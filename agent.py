import asyncio
import os
import sys

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.runner.livekit import generate_token_with_agent

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

SYSTEM_PROMPT = """You are AnswerBite, a friendly AI phone agent for restaurants.
You help customers with reservations, orders, and general questions.
Keep responses short and natural — this is a phone call.
Never mention you are an AI unless directly asked."""


async def main():
    livekit_url = os.environ["LIVEKIT_URL"]
    livekit_api_key = os.environ["LIVEKIT_API_KEY"]
    livekit_api_secret = os.environ["LIVEKIT_API_SECRET"]
    groq_api_key = os.environ["GROQ_API_KEY"]
    room_name = os.environ.get("LIVEKIT_ROOM_NAME", "answerbite-room")

    token = generate_token_with_agent(
        room_name, "AnswerBite-Agent", livekit_api_key, livekit_api_secret
    )

    transport = LiveKitTransport(
        url=livekit_url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = GroqSTTService(api_key=groq_api_key, model="whisper-large-v3-turbo")
    llm = GroqLLMService(api_key=groq_api_key, model="llama-3.1-8b-instant")
    tts = GroqTTSService(api_key=groq_api_key, voice="Celeste-PlayAI")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages=messages)
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        aggregator.user(),
        llm,
        tts,
        transport.output(),
        aggregator.assistant(),
    ])

    task = PipelineTask(pipeline)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant}")
        await task.queue_frames([aggregator.user().get_context_frame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
