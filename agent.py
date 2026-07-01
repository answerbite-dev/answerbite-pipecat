import asyncio
import os
import sys
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.groq import GroqLLMService, GroqSTTService, GroqTTSService
from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

SYSTEM_PROMPT = """You are AnswerBite, a friendly and efficient AI phone agent for restaurants. 
You help customers with reservations, orders, and general questions.
Keep responses concise and natural — this is a phone call.
If asked about hours, specials, or reservations, be helpful and conversational.
Never say you are an AI unless directly asked."""


async def main():
    livekit_url = os.environ.get("LIVEKIT_URL")
    livekit_api_key = os.environ.get("LIVEKIT_API_KEY")
    livekit_api_secret = os.environ.get("LIVEKIT_API_SECRET")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    room_name = os.environ.get("LIVEKIT_ROOM_NAME", "answerbite-room")

    transport = LiveKitTransport(
        url=livekit_url,
        token=None,
        room_name=room_name,
        params=LiveKitParams(
            api_key=livekit_api_key,
            api_secret=livekit_api_secret,
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = GroqSTTService(api_key=groq_api_key, model="whisper-large-v3-turbo")

    llm = GroqLLMService(api_key=groq_api_key, model="llama-3.1-8b-instant")

    tts = GroqTTSService(api_key=groq_api_key, voice="Celeste-PlayAI")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant}")
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
