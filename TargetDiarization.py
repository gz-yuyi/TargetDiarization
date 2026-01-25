# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2025-10-13

import io
import math
import modelscope
import numpy as np
import os
import sys
import torch
from pyannote.audio import Pipeline as pyannote_pipeline
from TargetASR import TargetASR
from typing import Union, Literal
from AudioProcessor import AudioProcessor
from dotenv import load_dotenv

# Add environment
file_dir = str(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
sys.path.append(file_dir)
load_dotenv()


# Main class
class TargetDiarization:
    def __init__(self,
                 diarization_pipeline_dir: str = "iic/speech_campplus_speaker-diarization_common", od_model_dir: str = "pyannote/speaker-diarization-3.1", mdx_weights_file: str = "mdx/weights/UVR-MDX-NET-Inst_HQ_3.onnx", embedding_model_dir: str = "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
                 vad_model_dir: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", asr_model_dir: str = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", separater_weights_folder: str = "checkpoints/mossformer2-finetune", restorer_weights_folder: str = "JusperLee/Apollo",
                 asr_engine: str = "paraformer", pyannote_clustering_threshold: float = 0.0, target_similarity_threshold: float = 0.0, silence_db_threshold: float = -45.0, vad_min_silence: float = 0.5, cuda_device: int = 0, verbose_log: bool = False, *args, **kwargs):
        self.file_dir = str(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
        self.diarization_pipeline_dir = diarization_pipeline_dir
        self.od_model_dir = od_model_dir
        self.pyannote_clustering_threshold = pyannote_clustering_threshold
        self.target_similarity_threshold = target_similarity_threshold
        self.silence_db_threshold = silence_db_threshold
        self.vad_min_silence = vad_min_silence
        self.asr_engine = asr_engine
        self.cuda_device = cuda_device
        self.verbose_log = verbose_log
        self.ap = AudioProcessor(cuda_device=self.cuda_device, verbose_log=False)
        self.tasr = TargetASR(cuda_device=self.cuda_device, verbose_log=self.verbose_log, embedding_model_dir=embedding_model_dir, vad_model_dir=vad_model_dir, asr_model_dir=asr_model_dir, separater_weights_folder=separater_weights_folder, restorer_weights_folder=restorer_weights_folder, mdx_weights_file=mdx_weights_file)
        self.get_device()
        self.load_model()

    # Get CUDA device ID (-1=CPU)
    def get_device(self):
        if self.cuda_device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = "cpu"
        else:
            if self.cuda_device == -1:
                self.device = "cpu"
            else:
                self.device = f"cuda:{self.cuda_device}"
                try:
                    torch.cuda.set_device(self.device)
                except Exception as e:
                    print(f"Failed to set CUDA device: {e}")

    # Load local models
    def load_model(self):
        def correct_path(path: str):
            path = path.replace("\\", "/").rstrip("/")
            if not os.path.isabs(path):
                path = f"{self.file_dir}/{path}"
            path = os.path.abspath(path)
            return path

        print(f"Init diarization pipeline from: {self.diarization_pipeline_dir}")
        self.diarization_pipeline_dir = correct_path(self.diarization_pipeline_dir)
        self.sd_pipeline = modelscope.pipelines.pipeline(
            task="speaker-diarization",
            model=self.diarization_pipeline_dir,
            disable_update=False
        )
        print(f"Init Overlap-detection model from: {self.od_model_dir}")
        if os.path.isabs(self.od_model_dir) and not self.od_model_dir.endswith("config.yaml"):
            self.od_model_dir = f"{self.od_model_dir}/config.yaml"
        os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        hf_token = os.getenv("HF_TOKEN")
        try:
            self.od_pipeline = pyannote_pipeline.from_pretrained(self.od_model_dir, use_auth_token=hf_token)
            self.od_pipeline.to(torch.device(self.device))
            if self.pyannote_clustering_threshold > 0.0:
                self.od_pipeline._pipelines['clustering']._instantiated['threshold'] = float(self.pyannote_clustering_threshold)
        except Exception as e:
            self.od_pipeline = None
            print("====================================================")
            print(f"Failed to init pyannote model from HuggingFace: {e}")
            if not hf_token:
                print("Please fill the `HF_TOKEN` field in .env config file.")
            print("!!! SYSTEM WILL SKIP VOICE OVERLAP DETECTION PART !!!")
            print("=====================================================")

    # Main method
    def infer(self, wav_file: Union[str, np.ndarray, io.BytesIO], target_file: Union[str, np.ndarray, io.BytesIO] = None, sampling_rate: int = 16000, is_single: bool = False, output_target_audio: bool = True):
        long_audio_threshold = 30.0
        if isinstance(wav_file, str) or isinstance(wav_file, io.BytesIO):
            audio_data, sampling_rate = self.ap.read_audio(wav_file)
        else:
            audio_data = wav_file
        audio_data, sampling_rate = self.audio_preprocess(audio_data=audio_data, sampling_rate=sampling_rate)
        target_embedding = None
        if target_file is not None:
            if isinstance(target_file, str) or isinstance(target_file, io.BytesIO):
                target_audio_data, target_sampling_rate = self.ap.read_audio(target_file)
            else:
                target_audio_data = target_file
            target_audio_data, target_sampling_rate = self.audio_preprocess(audio_data=target_audio_data, sampling_rate=target_sampling_rate)
            target_vad_result = self.tasr.asrp.vad_detection(wav_file=target_audio_data)
            if target_vad_result:
                start_time = target_vad_result[0][0]
                end_time = target_vad_result[-1][1]
                if end_time - start_time < 4.0:
                    print("WARNING: The valid speaking duration of target audio is less than 4s. This may cause a bad result.")
                target_audio_data = self.ap.split_audio_by_time(audio_data=target_audio_data, sampling_rate=target_sampling_rate, start_time=start_time, end_time=end_time)
                target_embedding = self.tasr.get_speaker_embedding(wav_file=target_audio_data)
            else:
                print("ERROR: No VAD result in target audio. Automatically select one speaker from the input audio as the target.")
        sd_result = None
        pyannote_result = None
        if audio_data.shape[0] / sampling_rate >= long_audio_threshold or self.od_pipeline is None:
            try:
                sd_result = self.sd_pipeline(audio_data)
                sd_result = self.sd_result_parser(sd_result=sd_result, is_single=is_single, combine_timerange=False)
            except Exception as e:
                sd_result = None
                print(e)
        if not sd_result and self.od_pipeline is not None:
            pyannote_result = self.od_pipeline({
                "waveform": self.ap.ndarray_to_torchaudio(audio_data=audio_data),
                "sample_rate": sampling_rate
            })
            sd_result = self.od_result_parser(od_result=pyannote_result, is_single=is_single, output_overlap=False)
        if self.verbose_log:
            print("sd_result:", sd_result)
        overlap_map = []
        target_spk = ""
        if not is_single:
            if pyannote_result is None and self.od_pipeline is not None:
                pyannote_result = self.od_pipeline({
                    "waveform": self.ap.ndarray_to_torchaudio(audio_data=audio_data),
                    "sample_rate": sampling_rate
                })
            od_result = self.od_result_parser(od_result=pyannote_result, sd_result=sd_result)
            if self.verbose_log:
                print("od_result:", od_result)
            sd_result, overlap_map = self.apply_od_result(sd_result=sd_result, od_result=od_result)
            if self.verbose_log:
                print("refined sd_result:", sd_result)
                print("overlap_map:", overlap_map)
            if target_embedding is not None:
                target_spk = self.target_embedding_to_target_spk(target_embedding=target_embedding, audio_data=audio_data, sampling_rate=sampling_rate, sd_result=sd_result, overlap_map=overlap_map)
            else:
                target_spk, target_embedding = self.sd_result_to_target_embedding(audio_data=audio_data, sampling_rate=sampling_rate, sd_result=sd_result, overlap_map=overlap_map)
            if self.verbose_log:
                print("target_spk:", target_spk)
        asr_result = self.sd_result_to_asr_audio(audio_data=audio_data, sampling_rate=sampling_rate, sd_result=sd_result, overlap_map=overlap_map, target_spk=target_spk, target_embedding=target_embedding)
        asr_result = self.recheck_target_speaker(result=asr_result, target_spk=target_spk, target_embedding=target_embedding)
        asr_result, target_audio_data = self.asr_audio_parser(asr_result=asr_result, target_spk=target_spk, output_target_audio=output_target_audio)
        return target_spk, asr_result, target_audio_data
    
    # Audio preprocess module chain
    def audio_preprocess(self, audio_data: np.ndarray, sampling_rate: int = 16000, stream_mode: bool = False, output_audio_only: bool = False):
        audio_data = self.ap.audio_to_mono(audio_data=audio_data)
        audio_data = self.ap.int16_to_float32(audio_data=audio_data)
        try:
            audio_data, sampling_rate = self.ap.audio_resample(audio_data=audio_data, orig_sr=sampling_rate, target_sr=16000)
            audio_data = self.ap.audio_loudness_control(audio_data=audio_data, sampling_rate=sampling_rate)
            if stream_mode:
                audio_data, _ = self.tasr.ap.separate_speaker(audio_data=audio_data, sampling_rate=sampling_rate)
            else:
                audio_data = self.tasr.ap.denoise_vocal(audio_data=audio_data, sampling_rate=sampling_rate)
            audio_data = self.ap.audio_loudness_control(audio_data=audio_data, sampling_rate=sampling_rate)
        except Exception as e:
            print(f"Failed in func audio_preprocess: {e}")
        if output_audio_only:
            return audio_data
        else:
            return audio_data, sampling_rate
    
    # Parse Modelscope diarization pipeline result
    def sd_result_parser(self, sd_result: dict, is_single: bool = False, combine_timerange: bool = False):
        result = {}
        if not sd_result or not sd_result['text']:
            return result
        timerange_speaker_list = sd_result['text']
        timerange_speaker_list = sorted(timerange_speaker_list, key=lambda item: item[0])
        prev_label = ""
        prev_start_point = 0.0
        prev_end_point = 0.0
        for timerange_speaker in timerange_speaker_list:
            # label = f"SPEAKER_{str(timerange_speaker[-1]).zfill(2)}"
            if is_single:
                label = "0"
            else:
                label = str(int(timerange_speaker[-1]))
            if combine_timerange:
                if not prev_label:
                    prev_label = label
                    prev_start_point = timerange_speaker[0]
                    prev_end_point = timerange_speaker[1]
                    continue
                if label == prev_label:
                    prev_end_point = timerange_speaker[1]
                    continue
                else:
                    start_point = prev_start_point
                    end_point = prev_end_point
                    prev_label = label
                    prev_start_point = timerange_speaker[0]
                    prev_end_point = timerange_speaker[1]
            else:
                start_point = timerange_speaker[0]
                end_point = timerange_speaker[1]
            if label not in result:
                result[label] = []
            result[label].append((round(start_point, 3), round(end_point, 3)))
        if prev_label and prev_label not in result:
            result[prev_label] = [(round(prev_start_point, 3), round(prev_end_point, 3))]
        if is_single and result:
            result['0'] = self.merge_timeranges(result['0'])
        return result
    
    # Parse Pyannote diarization result, and get overlapping regions
    def od_result_parser(self, od_result: dict, sd_result: dict = {}, is_single: bool = False, output_overlap: bool = True):
        result = {}
        if not od_result:
            return result
        for segment, track, label in od_result.itertracks(yield_label=True):
            if is_single:
                label = "0"
            else:
                label = str(int(label.split("_")[-1]))
            if label not in result:
                result[label] = []
            result[label].append((round(segment.start, 3), round(segment.end, 3)))
        if is_single:
            result['0'] = self.merge_timeranges(result['0'])
        if sd_result:
            result = self.sd_key_matcher(sd_result, result)
        if output_overlap:
            result = self.get_speaker_overlap(result)
        return result
    
    # Calculate IoU of one-to-one input
    def calc_single_iou(self, pred_duration: list, gt_duration: list):
        if len(pred_duration) != 2 or len(gt_duration) != 2:
            raise ValueError("Length of pred_duration and gt_duration should be 2.")
        if pred_duration[0] > pred_duration[1]:
            pred_duration = [pred_duration[1], pred_duration[0]]
        if gt_duration[0] > gt_duration[1]:
            gt_duration = [gt_duration[1], gt_duration[0]]
        if pred_duration[1] <= gt_duration[0] or gt_duration[1] <= pred_duration[0]:
            return 0.0
        intersection_start = max(pred_duration[0], gt_duration[0])
        intersection_end = min(pred_duration[1], gt_duration[1])
        inter_duration = intersection_end - intersection_start
        union_start = min(pred_duration[0], gt_duration[0])
        union_end = max(pred_duration[1], gt_duration[1])
        union_duration = union_end - union_start
        iou = inter_duration / union_duration
        return iou

    # Calculate average IoU of many-to-many input
    def calc_multi_iou(self, pred_durations: list, gt_durations: list, method: Literal["pred_to_gt", "gt_to_pred", "both_mean"] = "both_mean"):
        if len(pred_durations) == 0 or len(gt_durations) == 0:
            raise ValueError("Length of pred_durations and gt_durations cannot be zero.")
        pred_to_gt_ious = []
        for pred_duration in pred_durations:
            pred_gt_ious = []
            for gt_duration in gt_durations:
                iou = self.calc_single_iou(pred_duration=pred_duration, gt_duration=gt_duration)
                pred_gt_ious.append(iou)
            if pred_gt_ious:
                pred_to_gt_ious.append(max(pred_gt_ious))
            else:
                pred_to_gt_ious.append(0.0)
        gt_to_pred_ious = []
        for gt_duration in gt_durations:
            gt_pred_ious = []
            for pred_duration in pred_durations:
                iou = self.calc_single_iou(pred_duration=pred_duration, gt_duration=gt_duration)
                gt_pred_ious.append(iou)
            if gt_pred_ious:
                gt_to_pred_ious.append(max(gt_pred_ious))
            else:
                gt_to_pred_ious.append(0.0)
        pred_to_gt_iou = np.mean(pred_to_gt_ious) if pred_to_gt_ious else 0.0
        gt_to_pred_iou = np.mean(gt_to_pred_ious) if gt_to_pred_ious else 0.0
        average_iou = (pred_to_gt_iou + gt_to_pred_iou) / 2.0
        if method == "pred_to_gt":
            return pred_to_gt_iou
        elif method == "gt_to_pred":
            return gt_to_pred_iou
        else:
            return average_iou

    # Calculate average IoU of many-to-many input, with negative region punishment
    def calc_iou_score(self, pred_durations: list, gt_durations: list, positive_weight: float = 1.0, negative_weight: float = 1.0):
        def deduplicate(durations):
            unique = []
            for d in durations:
                if not any(u[0] == d[0] and u[1] == d[1] for u in unique):
                    unique.append(d)
            return sorted(unique, key=lambda x: x[0])

        if len(pred_durations) == 0 or len(gt_durations) == 0:
            raise ValueError("Length of pred_durations and gt_durations cannot be zero.")
        inside_durations = []
        outside_durations = []
        for gt_duration in gt_durations:
            for pred_duration in pred_durations:
                if pred_duration[0] >= gt_duration[0] and pred_duration[1] <= gt_duration[1]:
                    inside_durations.append(pred_duration)
                    break
                elif pred_duration[0] < gt_duration[0] < pred_duration[1]:
                    outside_durations.append([pred_duration[0], gt_duration[0]])
                    if gt_duration[0] < pred_duration[1] <= gt_duration[1]:
                        inside_durations.append([gt_duration[0], pred_duration[1]])
                    else:
                        inside_durations.append([gt_duration[0], gt_duration[1]])
                        outside_durations.append([gt_duration[1], pred_duration[1]])
                    break
                elif pred_duration[0] < gt_duration[1] < pred_duration[1]:
                    inside_durations.append([pred_duration[0], gt_duration[1]])
                    outside_durations.append([gt_duration[1], pred_duration[1]])
                    break
        for pred_duration in pred_durations:
            is_inside = False
            for gt_duration in gt_durations:
                if (pred_duration[0] < gt_duration[0] < pred_duration[1]) or (pred_duration[0] < gt_duration[1] < pred_duration[1]) or (gt_duration[0] <= pred_duration[0] and pred_duration[1] <= gt_duration[1]):
                    is_inside = True
                    break
            if not is_inside:
                outside_durations.append(pred_duration)
        inside_durations = deduplicate(inside_durations)
        outside_durations = deduplicate(outside_durations)
        positive_score = 0.0
        # for inside_duration in inside_durations:
        #     iou = calc_multi_iou(pred_durations=[inside_duration], gt_durations=gt_durations, method="pred_to_gt")
        #     positive_score = positive_score + iou
        # positive_score = positive_score / len(gt_durations)
        total_inside_length = sum(d[1] - d[0] for d in inside_durations)
        for inside_duration in inside_durations:
            length_ratio = (inside_duration[1] - inside_duration[0]) / total_inside_length
            iou = self.calc_multi_iou(pred_durations=[inside_duration], gt_durations=gt_durations, method="pred_to_gt")
            positive_score = positive_score + (iou * length_ratio)
            positive_score = positive_score + iou
        gt_duration_sum = 0.0
        for gt_duration in gt_durations:
            gt_duration_sum = gt_duration_sum + (gt_duration[1] - gt_duration[0])
        negative_score = 0.0
        for outside_duration in outside_durations:
            negative_score = negative_score + ((outside_duration[1] - outside_duration[0]) / gt_duration_sum)
        score = positive_score * positive_weight - negative_score * negative_weight
        if positive_weight == 0.0:
            score = abs(score)
        score = max(0.0, min(score, 1.0))
        return score
    
    # Match od_result's keys to sd_result's
    def sd_key_matcher(self, source_sd: dict, target_sd: dict):
        refined_target_sd = target_sd
        target_source_mapper = {}
        mapped_targets = []
        for source_spk in source_sd:
            max_iou = 0.0
            related_target = None
            for target_spk in target_sd:
                if target_spk in mapped_targets:
                    continue
                iou_score = self.calc_iou_score(pred_durations=source_sd[source_spk], gt_durations=target_sd[target_spk])
                if iou_score > max_iou:
                    max_iou = iou_score
                    related_target = target_spk
            if related_target:
                target_source_mapper[related_target] = source_spk
                mapped_targets.append(related_target)
        if target_source_mapper:
            refined_target_sd = {}
            added_targets = []
            for target_spk in target_source_mapper:
                source_spk = target_source_mapper[target_spk]
                refined_target_sd[source_spk] = target_sd[target_spk]
                added_targets.append(target_spk)
            for target_spk in target_sd:
                if target_spk not in added_targets and target_spk not in refined_target_sd.keys():
                    refined_target_sd[target_spk] = target_sd[target_spk]
        return refined_target_sd
    
    # eg. [(1, 3), (2, 6), (8, 10), (10, 11)] -> [(1, 6), (8, 11)]
    def merge_timeranges(self, timeranges: list):
        if not timeranges:
            return []
        timeranges.sort(key=lambda x: x[0])
        merged = [timeranges[0]]
        for i in range(1, len(timeranges)):
            current_start, current_end = timeranges[i]
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        return merged
    
    # eg. [(0, 10)], [(3, 5)] -> [(0, 3), (5, 10)]
    def subtract_timeranges(self, base_timeranges: list, sub_timeranges: list):
        if not sub_timeranges:
            return sub_timeranges
        sub_timeranges = self.merge_timeranges(sub_timeranges)
        output_timeranges = []
        for b_start, b_end in base_timeranges:
            current_start = b_start
            for s_start, s_end in sub_timeranges:
                if current_start >= s_end:
                    continue
                if b_end <= s_start:
                    break
                overlap_start = max(current_start, s_start)
                overlap_end = min(b_end, s_end)
                if overlap_start < overlap_end:
                    if overlap_start > current_start:
                        output_timeranges.append((current_start, overlap_start))
                    current_start = overlap_end
            if current_start < b_end:
                output_timeranges.append((current_start, b_end))  
        return output_timeranges
    
    # Split sd_result by od_result, and generate overlapping map
    def apply_od_result(self, sd_result: dict, od_result: dict = {}):
        # refined_result = {spk: [] for spk in sd_result}
        refined_result = {}
        overlap_map = []
        overlap_timeranges = []
        total_overlap_regions = []
        if not od_result:
            return sd_result, overlap_map
        for od_timeranges in od_result.values():
            total_overlap_regions.extend(od_timeranges)
        total_overlap_regions = self.merge_timeranges(total_overlap_regions)
        for speakers, overlap_timerange in od_result.items():
            spk_pair = speakers.split('-')
            # if not all(spk in refined_result for spk in spk_pair):
            #     continue
            for spk in spk_pair:
                # if spk in refined_result:
                #     refined_result[spk].extend(overlap_timerange)
                if spk not in refined_result:
                    refined_result[spk] = []
                refined_result[spk].extend(overlap_timerange)
            if overlap_timerange not in overlap_timeranges:
                overlap_timeranges.extend(overlap_timerange)
        for spk, timerange in sd_result.items():
            if not timerange:
                continue
            if spk not in refined_result:
                refined_result[spk] = []
            non_overlap_timerange = self.subtract_timeranges(timerange, total_overlap_regions)
            refined_result[spk].extend(non_overlap_timerange)
        for spk in refined_result:
            refined_result[spk].sort(key=lambda x: x[0])
        for overlap_timerange in overlap_timeranges:
            overlap_map.append([])
            for spk, timeranges in refined_result.items():
                for i in range(len(timeranges)):
                    if timeranges[i] == overlap_timerange:
                        overlap_map[-1].append((spk, i))
        overlap_map = [om for om in overlap_map if om]
        return refined_result, overlap_map
    
    # Get non-overlapping or overlapping regions from sd_result
    def subtract_overlap(self, sd_result: dict, overlap_map: list = [], reverse_output: bool = False):
        if not overlap_map:
            return sd_result
        result = {spk: [] for spk in sd_result}
        spk_overlap_idx = {spk: [] for spk in sd_result}
        for overlap_items in overlap_map:
            for overlap_item in overlap_items:
                spk, index = overlap_item
                if index not in spk_overlap_idx[spk]:
                    spk_overlap_idx[spk].append(index)
        for spk in sd_result:
            for i in range(len(sd_result[spk])):
                if reverse_output:
                    if i in spk_overlap_idx[spk]:
                        result[spk].append(sd_result[spk][i])
                else:
                    if i not in spk_overlap_idx[spk]:
                        result[spk].append(sd_result[spk][i])
        return result

    # Get number of speakers from sd_result / od_result
    def get_speaker_num(self, result: dict, threshold: float = 0.0):
        if len(result) == 1:
            return len(result)
        if threshold <= 0:
            return len(result)
        main_speaker = ""
        main_speaker_duration = 0
        for speaker, speaker_value in result.items():
            duration_sum = 0
            for r in speaker_value:
                duration_sum = duration_sum + (r[1] - r[0])
            if duration_sum > main_speaker_duration:
                main_speaker = speaker
        speaker_num = 0
        for speaker, speaker_value in result.items():
            if speaker == main_speaker:
                speaker_num = speaker_num + 1
                continue
            for r in speaker_value:
                if r[1] - r[0] > threshold:
                    speaker_num = speaker_num + 1
                    break
        return speaker_num

    # Get overlapping regions from od_result
    def get_speaker_overlap(self, result: dict, min_overlap_sec: float = 0.4):
        def get_overlap_between_2speakers(speaker_a: list, speaker_b: list):
            overlap_result = []
            for interval1 in speaker_a:
                start1, end1 = interval1[0], interval1[1]
                for interval2 in speaker_b:
                    start2, end2 = interval2[0], interval2[1]
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    if overlap_start < overlap_end and overlap_end - overlap_start >= min_overlap_sec:
                        overlap_result.append((overlap_start, overlap_end))
            return overlap_result
        
        overlap_timerange = {}
        if len(result) == 1:
            return overlap_timerange
        result_list = []
        for speaker in result:
            result_list.append(result[speaker])
        for i in range(len(result_list) - 1):
            for j in range(i + 1, len(result_list)):
                speaker_a_key = list(result.keys())[i]
                speaker_b_key = list(result.keys())[j]
                key = f"{speaker_a_key}-{speaker_b_key}"
                overlap_result = get_overlap_between_2speakers(result[speaker_a_key], result[speaker_b_key])
                if overlap_result:
                    overlap_timerange[key] = overlap_result
        return overlap_timerange
    
    # Get target speaker key and target embedding from sd_result
    def sd_result_to_target_embedding(self, audio_data: np.ndarray, sampling_rate: int = 16000, sd_result: dict = {}, overlap_map: list = [], target_spk: str = ""):
        if not sd_result:
            target_embedding = self.tasr.get_target_embedding(target_audio=audio_data, output_embedding_list=False)
            return "", target_embedding
        if not target_spk or target_spk not in sd_result.keys():
            target_spk = list(sd_result.keys())[0]
            if len(sd_result) > 1:
                target_spk_duration = sum([timerange[1] - timerange[0] for timerange in sd_result[target_spk]])
                for spk in sd_result:
                    current_spk_duration = sum([timerange[1] - timerange[0] for timerange in sd_result[spk]])
                    if current_spk_duration > target_spk_duration:
                        target_spk = spk
                        target_spk_duration = current_spk_duration
        if overlap_map:
            sd_result = self.subtract_overlap(sd_result=sd_result, overlap_map=overlap_map)
        target_timeranges = sd_result[target_spk]
        target_audio_data_list = []
        for timerange in target_timeranges:
            if timerange[1] - timerange[0] < 0.4:
                continue
            audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
            target_audio_data_list.append(audio_data)
        if target_audio_data_list:
            target_audio_data = np.concatenate(target_audio_data_list, axis=0)
            target_embedding = self.tasr.get_target_embedding(target_audio=target_audio_data, output_embedding_list=False)
        else:
            target_embedding = self.tasr.get_target_embedding(target_audio=audio_data, output_embedding_list=False)
        return target_spk, target_embedding
    
    # Get target speaker key from target embedding by calculating avg_score per speaker
    def target_embedding_to_target_spk(self, target_embedding: np.ndarray, audio_data: np.ndarray, sampling_rate: int = 16000, sd_result: dict = {}, overlap_map: list = []):
        target_spk = ""
        if not sd_result:
            return target_spk
        score_map = []
        sd_result = self.subtract_overlap(sd_result=sd_result, overlap_map=overlap_map)
        for spk in sd_result:
            score_list = []
            for timerange in sd_result[spk]:
                clip_audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
                clip_embedding = self.tasr.get_speaker_embedding(wav_file=clip_audio_data)
                score = self.tasr.cosine_similarity(embedding_a=target_embedding, embedding_b=clip_embedding)
                score_list.append(score)
            if score_list:
                avg_score = sum(score_list) / len(score_list)
                score_map.append([spk, avg_score])
        if score_map:
            score_map.sort(key=lambda x: x[1], reverse=True)
            target_spk = score_map[0][0]
        return target_spk
    
    # Recheck target speaker clips using target embedding (-1=uncertain)
    def recheck_target_speaker(self, result: list, target_spk: str, target_embedding: np.ndarray, method: Literal["recheck_target", "recheck_others", "recheck_both"] = "recheck_target"):
        if not result:
            return []
        for i in range(len(result)):
            result[i]['score'] = -1.0
        if target_embedding is None:
            return result
        if not self.target_similarity_threshold or self.target_similarity_threshold == 0.0:
            return result
        for i in range(len(result)):
            if method == "recheck_target" and result[i]['speaker'] != target_spk:
                continue
            if method == "recheck_others" and result[i]['speaker'] == target_spk:
                continue
            clip_audio_data = result[i].get("audio", None)
            if clip_audio_data is None:
                continue
            clip_embedding = self.tasr.get_speaker_embedding(wav_file=clip_audio_data)
            score = self.tasr.cosine_similarity(embedding_a=target_embedding, embedding_b=clip_embedding)
            result[i]['score'] = round(score, 3)
            if score >= self.target_similarity_threshold:
                if result[i]['speaker'] != target_spk:
                    result[i]['speaker'] = target_spk
            else:
                if result[i]['speaker'] == target_spk:
                    result[i]['speaker'] = "-1"
        return result
    
    # Get diarization ASR result
    def sd_result_to_asr_audio_legacy(self, audio_data: np.ndarray, sampling_rate: int = 16000, sd_result: dict = {}, overlap_map: list = [], target_spk: str = "", target_embedding: np.ndarray = None):
        asr_result = []
        more_args = {
            "asr_engine": self.asr_engine,
            "vad_model": "funasr",
            "no_punc": False,
            "preprocess": []
        }
        if not sd_result:
            return asr_result
        if overlap_map:
            sd_result_single = self.subtract_overlap(sd_result=sd_result, overlap_map=overlap_map)
            sd_result_overlap = self.subtract_overlap(sd_result=sd_result, overlap_map=overlap_map, reverse_output=True)
        else:
            sd_result_single = sd_result
            sd_result_overlap = {}
        for spk in sd_result_single:
            for timerange in sd_result_single[spk]:
                clip_audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
                clip_text = self.tasr.single_speaker_asr(asr_audio=clip_audio_data, more_args=more_args)[0]['text']
                if not clip_text:
                    continue
                asr_result.append({
                    "speaker": spk,
                    "timerange": timerange,
                    "text": clip_text,
                    "type": "single",
                    "audio": clip_audio_data
                })
        if not target_spk or target_embedding is None:
            for spk in sd_result_overlap:
                for timerange in sd_result_overlap[spk]:
                    clip_audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
                    clip_result_list = self.tasr.single_speaker_asr(asr_audio=clip_audio_data, more_args=more_args)
                    if not clip_result_list:
                        continue
                    clip_text = clip_result_list[0]['text'].strip()
                    if not clip_text:
                        continue
                    asr_result.append({
                        "speaker": spk,
                        "timerange": timerange,
                        "text": clip_text,
                        "type": "overlap",
                        "audio": clip_audio_data
                    })
        else:
            noise_spk_list = list(set(sd_result.keys()) - set([target_spk]))
            for spk in sd_result_overlap:
                if spk in noise_spk_list:
                    continue
                for timerange in sd_result_overlap[spk]:
                    clip_audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
                    clip_result_list = self.tasr.multi_speakers_separate_asr(asr_audio=clip_audio_data, target_embedding=target_embedding, threshold=0.0, more_args=more_args)
                    if not clip_result_list:
                        continue
                    target_text = clip_result_list[0]['text'].strip()
                    target_audio = clip_result_list[0]['audio']
                    if target_text:
                        asr_result.append({
                            "speaker": spk,
                            "timerange": timerange,
                            "text": target_text,
                            "type": "overlap",
                            "audio": target_audio
                        })
                    noise_text = ""
                    if len(clip_result_list) > 1:
                        noise_text = clip_result_list[1]['text'].strip()
                        noise_audio = clip_result_list[1]['audio']
                    if noise_text:
                        for noise_spk in noise_spk_list:
                            asr_result.append({
                                "speaker": noise_spk,
                                "timerange": timerange,
                                "text": noise_text,
                                "type": "overlap",
                                "audio": noise_audio
                            })
        if asr_result:
            asr_result.sort(key=lambda x: x['timerange'][0])
        return asr_result
    
    # Get diarization ASR result
    def sd_result_to_asr_audio(self, audio_data: np.ndarray, sampling_rate: int = 16000, sd_result: dict = {}, overlap_map: list = [], target_spk: str = "", target_embedding: np.ndarray = None):
        asr_result = []
        more_args = {
            "asr_engine": self.asr_engine,
            "vad_model": "funasr",
            "no_punc": False,
            "preprocess": []
        }
        def has_speech(audio_clip: np.ndarray) -> bool:
            if audio_clip is None or audio_clip.size == 0:
                return False
            if self.silence_db_threshold is not None:
                rms = np.sqrt(np.mean(np.square(audio_clip.astype(np.float32))))
                rms_db = 20.0 * np.log10(rms + 1e-9)
                if rms_db < self.silence_db_threshold:
                    return False
            try:
                vad = self.tasr.asrp.vad_detection(wav_file=audio_clip, min_silence_sec=self.vad_min_silence)
            except Exception as e:
                if self.verbose_log:
                    print(f"VAD detection failed, keep clip: {e}")
                return True
            return bool(vad)
        if not sd_result:
            return asr_result
        if overlap_map:
            sd_result_single = self.subtract_overlap(sd_result=sd_result, overlap_map=overlap_map)
            sd_result_overlap = self.subtract_overlap(sd_result=sd_result, overlap_map=overlap_map, reverse_output=True)
        else:
            sd_result_single = sd_result
            sd_result_overlap = {}
        for spk in sd_result_single:
            for timerange in sd_result_single[spk]:
                clip_audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
                if not has_speech(clip_audio_data):
                    continue
                asr_result.append({
                    "speaker": spk,
                    "timerange": timerange,
                    "text": "",
                    "type": "single",
                    "audio": clip_audio_data
                })
        if not target_spk or target_embedding is None:
            for spk in sd_result_overlap:
                for timerange in sd_result_overlap[spk]:
                    clip_audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
                    if not has_speech(clip_audio_data):
                        continue
                    asr_result.append({
                        "speaker": spk,
                        "timerange": timerange,
                        "text": "",
                        "type": "overlap",
                        "audio": clip_audio_data
                    })
        else:
            noise_spk_list = list(set(sd_result.keys()) - set([target_spk]))
            for spk in sd_result_overlap:
                if spk in noise_spk_list:
                    continue
                for timerange in sd_result_overlap[spk]:
                    clip_audio_data = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
                    if not has_speech(clip_audio_data):
                        continue
                    clip_result_list = self.tasr.multi_speakers_separate_asr(asr_audio=clip_audio_data, target_embedding=target_embedding, threshold=0.0, is_output_asr=False, more_args=more_args)
                    if not clip_result_list:
                        continue
                    target_audio = clip_result_list[0]['audio']
                    target_audio = self.ap.audio_loudness_control(audio_data=target_audio, sampling_rate=sampling_rate)
                    target_timerange = [round(timerange[0] + clip_result_list[0]['timerange'][0], 3), round(timerange[0] + clip_result_list[0]['timerange'][1], 3)]
                    asr_result.append({
                        "speaker": spk,
                        "timerange": target_timerange,
                        "text": "",
                        "type": "overlap",
                        "audio": target_audio
                    })
                    if noise_spk_list and len(clip_result_list) > 1:
                        noise_audio = clip_result_list[1]['audio']
                        noise_timerange = [round(timerange[0] + clip_result_list[1]['timerange'][0], 3), round(timerange[0] + clip_result_list[1]['timerange'][1], 3)]
                        asr_result.append({
                            "speaker": noise_spk_list[0],
                            "timerange": noise_timerange,
                            "text": "",
                            "type": "overlap",
                            "audio": noise_audio
                        })
        if not asr_result:
            return asr_result
        asr_result.sort(key=lambda x: x['timerange'][0])
        new_asr_result = []
        spk_list = list(set([item['speaker'] for item in asr_result]))
        for spk in spk_list:
            combined_spk_audio = self.combine_audio_chunks(asr_result=asr_result, speaker=spk)
            if combined_spk_audio is not None:
                combined_spk_asr = self.tasr.asrp.asr_detection(wav_file=combined_spk_audio, asr_engine=self.asr_engine)[0]
                if "timestamp" not in combined_spk_asr.keys() or not combined_spk_asr['timestamp']:
                    text = combined_spk_asr['text'].strip()
                    if not more_args['no_punc']:
                        text = self.tasr.asrp.punctuation_restore(text=text)
                    new_asr_result.append({
                        "speaker": spk,
                        "timerange": [asr_result[0]['timerange'][0], asr_result[-1]['timerange'][1]],
                        "text": combined_spk_asr['text'],
                        "type": "single",
                        "audio": combined_spk_audio
                    })
                else:
                    for chunk_item in asr_result:
                        if chunk_item['speaker'] == spk:
                            text = ""
                            for char_item in combined_spk_asr['timestamp']:
                                start_point = math.floor(chunk_item['timerange'][0] * 10) / 10
                                end_point = math.ceil(chunk_item['timerange'][1] * 10) / 10
                                if start_point <= char_item[-1][0] <= end_point:
                                    if combined_spk_asr['language'] in ["zh", "ja", "ko", "yue"]:
                                        text = text + char_item[0]
                                    else:
                                        text = text + " " + char_item[0]
                            if not more_args['no_punc']:
                                text = self.tasr.asrp.punctuation_restore(text=text)
                            chunk_item['text'] = text
                            new_asr_result.append(chunk_item)
        new_asr_result.sort(key=lambda x: x['timerange'][0])
        return new_asr_result
    
    def combine_audio_chunks(self, asr_result: list, speaker: str, sampling_rate: int = 16000):
        combined_audio_data = None
        if not asr_result:
            return combined_audio_data
        combined_audio_list = []
        cursor = 0.0
        for item in asr_result:
            if item['speaker'] == speaker:
                if cursor < item['timerange'][0]:
                    silence_frame = int((item['timerange'][0] - cursor) * sampling_rate)
                    silence_clip = np.zeros(silence_frame, dtype=np.float32)
                    combined_audio_list.append(silence_clip)
                combined_audio_list.append(item['audio'])
                cursor = item['timerange'][1]
        if combined_audio_list:
            combined_audio_data = np.concatenate(combined_audio_list, axis=0)
        return combined_audio_data
    
    # Warp diarization ASR result
    def asr_audio_parser(self, asr_result: list, target_spk: str, output_target_audio: bool = True):
        result = []
        if not asr_result:
            return result, None
        if isinstance(asr_result, dict):
            asr_result = [asr_result]
        if not output_target_audio:
            for item in asr_result:
                item.pop("audio", None)
                result.append(item)
            return result, None
        target_audio_list = []
        target_audio_data = None
        asr_result.sort(key=lambda x: x['timerange'][0])
        cursor = 0.0
        sampling_rate = 16000
        for item in asr_result:
            if item['speaker'] == target_spk:
                silence_frame = int((item['timerange'][0] - cursor) * sampling_rate)
                if silence_frame > 0:
                    silence_clip = np.zeros(silence_frame, dtype=np.float32)
                    target_audio_list.append(silence_clip)
                target_audio_list.append(item['audio'].astype(np.float32))
                cursor = item['timerange'][1]
            item.pop("audio", None)
            result.append(item)
        if cursor < asr_result[-1]['timerange'][1]:
            silence_frame = int((asr_result[-1]['timerange'][1] - cursor) * sampling_rate)
            silence_clip = np.zeros(silence_frame, dtype=np.float32)
            target_audio_list.append(silence_clip)
        if target_audio_list:
            target_audio_data = np.concatenate(target_audio_list, axis=0)
        return result, target_audio_data
