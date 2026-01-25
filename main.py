# -*- coding: utf-8 -*-
# Written by GD Studio & Claude 4.0
# Date: 2025-9-19

import asyncio
import base64
import logging
import numpy as np
import os
import queue
import tempfile
import threading
import time
import traceback
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from TargetDiarizationStream import TargetDiarizationStream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Target Diarization API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
tds_model = None


# Response models
class DiarizationResult(BaseModel):
    speaker: str
    speaker_type: str
    timerange: List[float]
    text: str
    type: str
    score: float


class DiarizationResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


# Utility function: speaker_id to show_name
def format_speaker_info(speaker_id: str, target_speaker_id: str):
    if speaker_id == target_speaker_id:
        return "target"
    elif speaker_id == "-1":
        return "uncertain"
    else:
        return "other"


# Utility function: ndarray to base64
def audio_to_base64(audio_data: np.ndarray):
    if audio_data is None:
        return ""
    if audio_data.dtype == np.float32:
        audio_data = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_data.tobytes()
    return base64.b64encode(audio_bytes).decode('utf-8')


# Utility function: save uploaded file to temporary location
async def save_upload_file(upload_file: UploadFile):
    suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await upload_file.read()
        tmp_file.write(content)
        return tmp_file.name


# Utility function: delete temporary file
def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")


# Init models on startup
@app.on_event("startup")
async def startup_event():
    global tds_model
    logger.info("Init TargetDiarization...")
    load_dotenv()
    tds_args = {
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
        "is_vad_buffer": False if os.environ.get("IS_VAD_BUFFER") == "0" else True,
        "max_buffer_duration": float(os.environ.get("MAX_BUFFER_DURATION")) if os.environ.get("MAX_BUFFER_DURATION") is not None else None,
        "vad_min_silence": float(os.environ.get("VAD_MIN_SILENCE")) if os.environ.get("VAD_MIN_SILENCE") is not None else None,
        "use_asr_prompt": False if os.environ.get("USE_ASR_PROMPT") == "0" else True,
        "similarity_threshold": float(os.environ.get("SIMILARITY_THRESHOLD")) if os.environ.get("SIMILARITY_THRESHOLD") is not None else None,
        "loudness_diff_threshold": float(os.environ.get("LOUDNESS_DIFF_THRESHOLD")) if os.environ.get("LOUDNESS_DIFF_THRESHOLD") is not None else None
    }
    tds_args_refined = {}
    for key, value in tds_args.items():
        if value is not None:
            tds_args_refined[key] = value
    try:
        print("Init args:", tds_args_refined)
        tds_model = TargetDiarizationStream(**tds_args_refined)
        logger.info("TargetDiarization inited successfully")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to init TargetDiarization: {e}")
        raise


# API root endpoint
@app.get("/")
async def root():
    return {
        "message": "Target Diarization API",
        "version": "1.0.0",
        "endpoints": {
            "inference": "/diarization/infer",
            "streaming": "/diarization/stream",
            "health": "/health"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": tds_model is not None,
        "timestamp": int(time.time())
    }


# Non-streaming speaker diarization and ASR inference
@app.post("/diarization/infer")
async def diarization_infer(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    target_file: Optional[UploadFile] = File(None),
    sampling_rate: int = 16000,
    is_single: bool = False,
    output_target_audio: bool = True
):
    start_time = time.time()
    audio_path = None
    target_path = None
    try:
        if tds_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        audio_path = await save_upload_file(audio_file)
        target_file_path = None
        if target_file is not None:
            target_path = await save_upload_file(target_file)
            target_file_path = target_path
        logger.info(f"Starting inference: audio={audio_file.filename}, target={target_file.filename if target_file else None}")
        target_spk, final_result, target_audio_data = tds_model.infer(
            wav_file=audio_path,
            target_file=target_file_path,
            sampling_rate=sampling_rate,
            is_single=is_single,
            output_target_audio=output_target_audio
        )
        formatted_results = []
        for result in final_result:
            speaker_type = format_speaker_info(result['speaker'], target_spk)
            formatted_result = DiarizationResult(
                speaker=result['speaker'],
                speaker_type=speaker_type,
                timerange=list(result['timerange']),
                text=result['text'],
                type=result['type'],
                score=result.get("score", -1.0)
            )
            formatted_results.append(formatted_result)
        response_data = {
            "target_speaker_id": target_spk,
            "total_speakers": len(set(r['speaker'] for r in final_result if r['speaker'] != "-1")),
            "results": [result.dict() for result in formatted_results],
            "statistics": {
                "total_duration": round(max(r['timerange'][1] for r in final_result) if final_result else 0.0, 3),
                "target_speaker_duration": round(sum(r['timerange'][1] - r['timerange'][0] for r in final_result if r['speaker'] == target_spk), 3),
                "other_speakers_duration": round(sum(r['timerange'][1] - r['timerange'][0] for r in final_result if r['speaker'] != target_spk and r['speaker'] != "-1"), 3)
            }
        }
        if output_target_audio and target_audio_data is not None:
            response_data["target_audio_base64"] = audio_to_base64(target_audio_data)
        processing_time = time.time() - start_time
        if audio_path:
            background_tasks.add_task(cleanup_file, audio_path)
        if target_path:
            background_tasks.add_task(cleanup_file, target_path)
        logger.info(f"Inference completed in {processing_time:.2f}s")
        return DiarizationResponse(
            success=True,
            data=response_data,
            processing_time=round(processing_time, 3)
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Inference error: {str(e)}")
        if audio_path:
            background_tasks.add_task(cleanup_file, audio_path)
        if target_path:
            background_tasks.add_task(cleanup_file, target_path)
        return DiarizationResponse(
            success=False,
            error=f"Inference failed: {str(e)}",
            processing_time=round(time.time() - start_time, 3)
        )


# Streaming speaker diarization and ASR service
@app.websocket("/diarization/stream")
async def diarization_stream(websocket: WebSocket):
    await websocket.accept()
    target_audio_array = None
    try:
        if tds_model is None:
            await websocket.send_json({"type": "error", "message": "Model not loaded"})
            return
        config_message = await websocket.receive_json()
        config_data = config_message.get("data", {})
        if config_data.get("has_target_file", False):
            target_message = await websocket.receive_json()
            if target_message.get("type") == "target_audio":
                target_audio_base64 = target_message.get("data")
                target_audio_bytes = base64.b64decode(target_audio_base64)
                target_audio_array = np.frombuffer(target_audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        await websocket.send_json({
            "type": "config_ack",
            "data": {"config": config_data, "target_file_loaded": target_audio_array is not None}
        })
        audio_generator = audio_stream_generator(websocket)
        async for target_spk, asr_result, target_audio_data in async_infer_stream(audio_generator, target_audio_array, config_data):
            for segment in asr_result:
                await websocket.send_json({
                    "type": "segment_result",
                    "data": {
                        "target_speaker_id": target_spk,
                        "segment": {
                            "speaker": segment['speaker'],
                            "speaker_type": format_speaker_info(segment['speaker'], target_spk),
                            "timerange": segment['timerange'],
                            "text": segment['text'],
                            "type": segment['type']
                        }
                    }
                })
        await websocket.send_json({"type": "status", "message": "completed"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": f"Processing error: {str(e)}"})
        except:
            pass


# Generate audio chunks from WebSocket stream
async def audio_stream_generator(websocket: WebSocket):
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "audio_chunk":
                audio_base64 = message.get("data")
                audio_bytes = base64.b64decode(audio_base64)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                yield audio_array
            elif message.get("type") == "audio_end":
                break
    except WebSocketDisconnect:
        return


# Asynchronous streaming inference handler
async def async_infer_stream(audio_generator, target_file, config: Dict):
    audio_queue = queue.Queue()
    finished = threading.Event()
    
    # Collect audio chunks from async generator
    async def audio_collector():
        try:
            async for audio_chunk in audio_generator:
                audio_queue.put(audio_chunk)
            audio_queue.put(None)  # End marker
        except Exception as e:
            logger.error(f"Audio collection error: {e}")
            audio_queue.put(None)
        finally:
            finished.set()
    
    # Synchronous generator for model inference
    def sync_audio_generator():
        while True:
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
                if audio_chunk is None:  # End marker
                    break
                yield audio_chunk
            except queue.Empty:
                if finished.is_set() and audio_queue.empty():
                    break
                continue
    
    # Run inference in thread pool
    def run_inference_with_queue():
        try:
            for result in tds_model.infer_stream(
                audio_stream_generator=sync_audio_generator(),
                target_file=target_file,
                sampling_rate=config.get("sampling_rate", 16000),
                is_single=config.get("is_single", False),
                output_target_audio=config.get("output_target_audio", False)
            ):
                asyncio.run_coroutine_threadsafe(
                    result_queue.put(result), loop
                ).result()
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Inference thread error: {e}")
            asyncio.run_coroutine_threadsafe(
                result_queue.put(StopIteration), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(
                inference_finished.set(), loop
            ).result()
    
    collector_task = asyncio.create_task(audio_collector())
    try:
        loop = asyncio.get_event_loop()
        result_queue = asyncio.Queue()
        inference_finished = asyncio.Event()
        with ThreadPoolExecutor() as executor:
            inference_future = executor.submit(run_inference_with_queue)
            while True:
                try:
                    result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                    if result is StopIteration:
                        break
                    yield result
                except asyncio.TimeoutError:
                    if inference_finished.is_set():
                        break
                    continue
                except Exception as e:
                    logger.error(f"Result reading error: {e}")
                    break
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Stream inference error: {e}")
        raise
    finally:
        if not collector_task.done():
            collector_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
