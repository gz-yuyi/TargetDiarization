import gradio as gr
import json
import os
import urllib.request
import uuid
from dotenv import load_dotenv
from TargetDiarization import TargetDiarization

load_dotenv()
td_args = {
    "verbose_log": True if os.environ.get("VERBOSE_LOG") == "1" else False,
    "cuda_device": int(os.environ.get("CUDA_DEVICE")) if os.environ.get("CUDA_DEVICE") is not None else None,
    "target_similarity_threshold": float(os.environ.get("TARGET_SIMILARITY_THRESHOLD")) if os.environ.get("TARGET_SIMILARITY_THRESHOLD") is not None else None,
    "pyannote_clustering_threshold": float(os.environ.get("PYANNOTE_CLUSTERING_THRESHOLD")) if os.environ.get("PYANNOTE_CLUSTERING_THRESHOLD") is not None else None,
    "diarization_pipeline_dir": os.environ.get("DIARIZATION_PIPELINE_DIR"),
    "od_model_dir": os.environ.get("OD_MODEL_DIR"),
    "mdx_weights_file": os.environ.get("MDX_WEIGHTS_FILE"),
    "embedding_model_dir": os.environ.get("EMBEDDING_MODEL_DIR"),
    "asr_model_dir": os.environ.get("ASR_MODEL_DIR"),
    "vad_model_dir": os.environ.get("VAD_MODEL_DIR"),
    "separater_weights_folder": os.environ.get("SEPARATER_WEIGHTS_FOLDER"),
    "restorer_weights_folder": os.environ.get("RESTORER_WEIGHTS_FOLDER"),
    "silence_db_threshold": float(os.environ.get("SILENCE_DB_THRESHOLD")) if os.environ.get("SILENCE_DB_THRESHOLD") is not None else None,
    "vad_min_silence": float(os.environ.get("VAD_MIN_SILENCE")) if os.environ.get("VAD_MIN_SILENCE") is not None else None
}
td_args_refined = {}
for key, value in td_args.items():
    if value is not None:
        td_args_refined[key] = value
print("Init args:", td_args)
td = TargetDiarization(**td_args_refined)


def process_audio(asr_input, target_input, is_single, no_target_audio):
    if asr_input.startswith("http"):
        asr_audio = f"/tmp/{uuid.uuid4().hex}.wav"
        urllib.request.urlretrieve(asr_input, asr_audio)
    else:
        asr_audio = asr_input
    target_audio = None
    if not no_target_audio:
        if target_input and target_input.startswith("http"):
            target_audio = f"/tmp/{uuid.uuid4().hex}.wav"
            urllib.request.urlretrieve(target_input, target_audio)
        else:
            target_audio = target_input
    target_spk, asr_result, target_audio_data = td.infer(wav_file=asr_audio, target_file=target_audio, is_single=is_single)
    asr_result = str(json.dumps(asr_result, ensure_ascii=False, indent=2))
    target_audio_data = (16000, target_audio_data)
    return target_spk, asr_result, target_audio_data


with gr.Blocks(title="目标说话人日志") as demo:
    gr.Markdown("# 目标说话人日志")
    gr.Markdown("`webui.py`仅提供非流式版演示Demo，完整体验请移步`demo.html`")
    gr.Markdown("最后更新：2025.9.24")
    with gr.Tabs():
        with gr.TabItem("上传音频"):
            with gr.Row():
                with gr.Column():
                    target_audio_input = gr.Audio(type="filepath", label="上传目标说话人音频")
                    no_target_audio_1 = gr.Checkbox(label="从待检测音频判断目标说话人", value=False)
                with gr.Column():
                    asr_audio_input = gr.Audio(type="filepath", label="上传待检测音频")
                    is_single_1 = gr.Checkbox(label="待检测音频为单人", value=False)
            upload_button = gr.Button("处理上传音频")
        with gr.TabItem("音频URL"):
            with gr.Column():
                target_url_input = gr.Textbox(label="目标说话人音频URL", placeholder="https://example.com/audio.wav")
                no_target_audio_2 = gr.Checkbox(label="从待检测音频判断目标说话人", value=False)
            with gr.Column():
                asr_url_input = gr.Textbox(label="待检测音频URL", placeholder="https://example.com/audio.wav")
                is_single_2 = gr.Checkbox(label="待检测音频为单人", value=False)
            url_button = gr.Button("处理URL音频")
    with gr.Row():
        asr_result = gr.Textbox(label="说话人日志", lines=8)
        with gr.Column():
            target_spk = gr.Textbox(label="目标说话人", lines=1)
            target_audio_data = gr.Audio(label="目标说话人音频")
    upload_button.click(
        process_audio,
        inputs=[asr_audio_input, target_audio_input, is_single_1, no_target_audio_1],
        outputs=[target_spk, asr_result, target_audio_data]
    )
    url_button.click(
        process_audio,
        inputs=[asr_url_input, target_url_input, is_single_2, no_target_audio_2],
        outputs=[target_spk, asr_result, target_audio_data]
    )


if __name__ == "__main__":
    demo.launch(server_port=8300, root_path="/target-diarization")
