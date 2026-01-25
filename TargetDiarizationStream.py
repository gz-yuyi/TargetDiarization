# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2025-9-25

import io
import numpy as np
import re
from typing import Union, Generator
from silero_vad import load_silero_vad, get_speech_timestamps
from TargetDiarization import TargetDiarization


class TargetDiarizationStream(TargetDiarization):
    def __init__(self, is_vad_buffer: bool = True, use_asr_prompt: bool = False, similarity_threshold: float = 0.4, vad_min_silence: float = 0.3, max_buffer_duration: float = 30.0, loudness_diff_threshold: float = 12.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_vad_buffer = is_vad_buffer
        self.use_asr_prompt = use_asr_prompt
        self.similarity_threshold = similarity_threshold
        self.max_buffer_duration = max_buffer_duration
        self.vad_min_silence = vad_min_silence
        self.loudness_diff_threshold = loudness_diff_threshold

        self.current_time = 0.0
        self.target_embedding = None
        self.prev_asr_text = ""
        self.vad_buffer = []
        self.current_buffer_duration = 0.0
        self.system_loudness_diff = 0.0
        self.silero_vad = load_silero_vad()
    
    # Clear VAD buffer
    def clear_vad_buffer(self):
        self.vad_buffer.clear()
        self.current_buffer_duration = 0.0
    
    # Audio chunk preprocess module chain
    def chunk_preprocess(self, audio_data: np.ndarray, sampling_rate: int):
        audio_data = self.ap.audio_to_mono(audio_data=audio_data)
        audio_data = self.ap.int16_to_float32(audio_data=audio_data)
        audio_data, sampling_rate = self.ap.audio_resample(audio_data=audio_data, orig_sr=sampling_rate, target_sr=16000)
        return audio_data

    # Main method
    def infer_stream(self, audio_stream_generator: Generator, target_file: Union[str, np.ndarray, io.BytesIO, None] = None, sampling_rate: int = 16000, is_single: bool = False, output_target_audio: bool = False):
        self.current_time = 0.0
        self.clear_vad_buffer()
        if target_file is not None:
            if not isinstance(target_file, np.ndarray):
                target_audio_data, sampling_rate = self.ap.read_audio(file_path=target_file)
            else:
                target_audio_data = target_file.copy()
            if target_audio_data.shape[0] / sampling_rate >= 1.0:
                target_loudness = self.ap.meter_loudness(audio_data=target_audio_data, sampling_rate=sampling_rate)
                self.system_loudness_diff = target_loudness + 23.0
                target_audio_data = self.audio_preprocess(audio_data=target_audio_data, sampling_rate=sampling_rate, stream_mode=True, output_audio_only=True)
                target_vad_result = self.tasr.asrp.vad_detection(wav_file=target_audio_data)
                if target_vad_result:
                    start_time = target_vad_result[0][0]
                    end_time = target_vad_result[-1][1]
                    if end_time - start_time < 4.0:
                        print("WARNING: The valid speaking duration of target audio is less than 4s. This may cause a bad result.")
                    target_audio_data = self.ap.split_audio_by_time(audio_data=target_audio_data, sampling_rate=sampling_rate, start_time=start_time, end_time=end_time)
                self.target_embedding = self.tasr.get_target_embedding(target_audio=target_audio_data, output_embedding_list=False)
        try:
            for pcm_chunk in audio_stream_generator:
                pcm_chunk = self.chunk_preprocess(audio_data=pcm_chunk, sampling_rate=sampling_rate)
                for result in self.process_vad_chunk(pcm_chunk, is_single):
                    target_spk = "1"
                    asr_result, target_audio_data = self.asr_audio_parser(asr_result=result, target_spk=target_spk, output_target_audio=output_target_audio)
                    yield target_spk, asr_result, target_audio_data
        finally:
            if self.vad_buffer:
                combined_audio = np.concatenate(self.vad_buffer)
                for result in self.process_single_chunk(combined_audio, is_single):
                    target_spk = "1"
                    asr_result, target_audio_data = self.asr_audio_parser(asr_result=result, target_spk=target_spk, output_target_audio=output_target_audio)
                    yield target_spk, asr_result, target_audio_data
                self.clear_vad_buffer()

    # VAD buffer router
    def process_vad_chunk(self, pcm_chunk: np.ndarray, is_single: bool):
        if pcm_chunk is None or pcm_chunk.shape[0] == 0:
            return
        is_silence = False
        if self.system_loudness_diff != 0.0:
            pcm_loudness = self.ap.meter_loudness(audio_data=pcm_chunk, sampling_rate=16000)
            if pcm_loudness < -23.0 + self.system_loudness_diff - self.loudness_diff_threshold:
                is_silence = True
                pcm_chunk = np.full_like(pcm_chunk, fill_value=0.00001, dtype=np.float32)
            print(f"PCM loudness: {pcm_loudness} | {-23.0 + self.system_loudness_diff - self.loudness_diff_threshold}")
        self.vad_buffer.append(pcm_chunk)
        chunk_duration = round(pcm_chunk.shape[0] / 16000, 3)
        self.current_buffer_duration = self.current_buffer_duration + chunk_duration
        if not self.is_vad_buffer:
            if is_silence:
                return
            current_chunk = self.vad_buffer[-1]
            for result in self.process_single_chunk(pcm_chunk=current_chunk, is_single=is_single):
                yield result
            self.clear_vad_buffer()
            return
        if self.should_wait_for_next_chunk(is_silence=is_silence):
            return
        combined_audio = np.concatenate(self.vad_buffer)
        for result in self.process_single_chunk(pcm_chunk=combined_audio, is_single=is_single):
            yield result
        self.clear_vad_buffer()

    # Verify if system should wait for next chunk by max_buffer_duration, embedding similarity and VAD result of last chunk
    def should_wait_for_next_chunk(self, is_silence: bool = False):
        # Check if current chunk have a silence gap at the end
        def check_silence_gap(audio_data: np.ndarray, vad_result: list):
            if not vad_result:
                return True
            audio_duration = len(audio_data) / 16000
            last_speech_end = vad_result[-1][-1]
            silence_duration = audio_duration - last_speech_end
            return silence_duration >= self.vad_min_silence
        
        # Rule 1: check buffer duration
        if self.current_buffer_duration >= self.max_buffer_duration:
            print("Buffer duration exceeds max_buffer_duration, processing buffer")
            return False
        if not self.vad_buffer:
            print("No buffer, continue waiting")
            return True
        # Rule 2: process silence chunk
        combined_audio = np.concatenate(self.vad_buffer)
        audio_tensor = self.ap.ndarray_to_torchaudio(audio_data=combined_audio)
        res = get_speech_timestamps(audio=audio_tensor, model=self.silero_vad, threshold=0.5, min_silence_duration_ms=100, return_seconds=True)
        vad_result = [[clip['start'], clip['end']] for clip in res]
        print("silero_vad:", vad_result)
        chunk_vad_result = self.tasr.asrp.vad_detection(wav_file=self.vad_buffer[-1])
        if is_silence:
            if check_silence_gap(combined_audio, vad_result):
                print("Silence detected with sufficient gap, processing buffer")
                return False
            else:
                print("Silence detected but gap insufficient, continue waiting")
                return True
        # Rule 3: process non-silence chunk
        if not chunk_vad_result:
            print("No valid speech detected in current chunk, processing to clear")
            pcm_chunk = np.full_like(self.vad_buffer[-1], fill_value=0.00001, dtype=np.float32)
            self.vad_buffer.pop()
            self.vad_buffer.append(pcm_chunk)
            if len(self.vad_buffer) == 1:
                return True
            else:
                # return False
                return True
        if check_silence_gap(combined_audio, vad_result):
            print("Speech appears complete, processing buffer")
            return False
        # Rule 4: check speaker similarity
        if len(self.vad_buffer) > 1:
            current_chunk = self.vad_buffer[-1]
            prev_chunks = self.vad_buffer[:-1]
            prev_audio = np.concatenate(prev_chunks)
            prev_embedding = self.tasr.get_speaker_embedding(wav_file=prev_audio)
            current_embedding = self.tasr.get_speaker_embedding(wav_file=current_chunk)
            is_same = self.tasr.is_same_person(existed_embeddings=prev_embedding, target_embedding=current_embedding, threshold=self.similarity_threshold)
            if is_same:
                print("Same speaker detected, continue waiting")
                return True
            else:
                print("Different speaker detected, processing buffer")
                return False
        # Rule 5: default wait
        print("Default wait")
        return True

    # Audio chunk preprocess, diarization, ASR
    def process_single_chunk(self, pcm_chunk: np.ndarray, is_single: bool):
        pyannote_result = self.od_pipeline({
            "waveform": self.ap.ndarray_to_torchaudio(audio_data=pcm_chunk),
            "sample_rate": 16000
        })
        od_result = self.od_result_parser(od_result=pyannote_result, is_single=is_single, output_overlap=True)
        print("od_result:", od_result)
        is_overlap = True if od_result else False
        result = self.asr_audio_streaming(audio_data=pcm_chunk, is_overlap=is_overlap)
        print("ASR result:", result)
        if result is not None:
            self.prev_asr_text = result['text']
            yield result
    
    # Get diarization ASR result (streaming version)
    def asr_audio_streaming(self, audio_data: np.ndarray, is_overlap: bool = False, is_output_audio: bool = False):
        def remove_punc(detect_text: str):
            if not detect_text:
                return detect_text
            punc_pattern = r"[^\w\s]"
            detect_text = re.sub(punc_pattern, "", detect_text)
            detect_text = detect_text.lower().strip()
            return detect_text

        audio_duration = round(audio_data.shape[0] / 16000, 3)
        if audio_duration < 0.4:
            return None
        timerange = [self.current_time, self.current_time + audio_duration]
        self.current_time = self.current_time + audio_duration
        more_args = {
            "asr_engine": "paraformer",
            "no_punc": False,
            "preprocess": []
        }
        if self.use_asr_prompt and self.prev_asr_text:
            more_args.update({
                "prompt": self.prev_asr_text
            })
        audio_data = self.audio_preprocess(audio_data=audio_data, sampling_rate=16000, stream_mode=True, output_audio_only=True)
        vad_result = self.tasr.asrp.vad_detection(audio_data)
        if not vad_result:
            return None
        if self.target_embedding is None:
            target_loudness = self.ap.meter_loudness(audio_data=audio_data, sampling_rate=16000)
            self.system_loudness_diff = target_loudness + 23.0
            self.target_embedding = self.tasr.get_speaker_embedding(wav_file=audio_data)
            is_overlap = False
        pcm_loudness = self.ap.meter_loudness(audio_data=audio_data, sampling_rate=16000)
        if pcm_loudness < -23.0 + self.system_loudness_diff - self.loudness_diff_threshold:
            return None

        # import time
        # filename = str(int(time.time())) + "_processed.wav"
        # output_file = "BAK/temp/" + filename
        # self.ap.write_to_file(output_file, audio_data, 16000)
        # print("FunASR VAD:", vad_result)
        # print(f"File saved: {filename}")

        if is_overlap:
            clip_result_list = self.tasr.multi_speakers_separate_asr(asr_audio=audio_data, target_embedding=self.target_embedding, more_args=more_args, is_output_audio=True)
        else:
            clip_result_list = self.tasr.single_speaker_asr(asr_audio=audio_data, more_args=more_args)
        if not clip_result_list:
            return None
        if len(clip_result_list) > 1:
            clip_result_list = sorted(clip_result_list, key=lambda x: len(remove_punc(x['text'])), reverse=True)
        clip_text = clip_result_list[0]['text'].strip()
        if not clip_text:
            return None
        timerange = [self.current_time + vad_result[0][0], self.current_time + vad_result[-1][-1]]
        segment_audio = clip_result_list[0]['audio'] if is_overlap else audio_data
        segment_embedding = self.tasr.get_speaker_embedding(wav_file=segment_audio)
        is_target = self.tasr.is_same_person(existed_embeddings=segment_embedding, target_embedding=self.target_embedding, threshold=self.similarity_threshold)
        speaker_label = "1" if is_target else "0"
        type_label = "overlap" if is_overlap else "single"
        asr_result = {
            "speaker": speaker_label,
            "timerange": timerange,
            "text": clip_text,
            "type": type_label,
            "audio": segment_audio if is_output_audio else None
        }
        return asr_result
