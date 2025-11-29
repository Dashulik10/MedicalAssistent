"""Клиент для генерации ответов через Groq API."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq

from configuration.settings import settings

from .prompts import MEDICAL_ASSISTANT_SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Результат генерации ответа."""

    text: str
    model: str
    tokens_used: int
    generation_time: float


class GroqLLMClient:
    """
    Клиент для генерации ответов через Groq API.
    Использует модель openai/gpt-oss-120b.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Инициализация клиента.

        :param model_name: имя модели (по умолчанию из settings)
        :param temperature: температура генерации
        :param max_tokens: максимальное количество токенов
        """
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = model_name or settings.generation_model_name
        self.temperature = (
            temperature if temperature is not None else settings.generation_temperature
        )
        self.max_tokens = max_tokens or settings.generation_max_tokens

        logger.info(
            f"GroqLLMClient инициализирован: model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Генерирует ответ на основе контекста и запроса.

        :param query: вопрос пользователя
        :param context: агрегированные данные пациента
        :param system_prompt: системный промпт (по умолчанию медицинский ассистент)
        :return: результат генерации
        """
        system = system_prompt or MEDICAL_ASSISTANT_SYSTEM_PROMPT
        user_message = build_user_prompt(query, context)

        logger.info(
            f"Генерация ответа: query длина={len(query)}, context длина={len(context)}"
        )

        start_time = time.time()
        max_attempts = 2

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                generation_time = time.time() - start_time
                result_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0

                logger.info(
                    f"Ответ сгенерирован: {len(result_text)} символов, "
                    f"{tokens_used} токенов, {generation_time:.2f}с"
                )

                return GenerationResult(
                    text=result_text,
                    model=self.model,
                    tokens_used=tokens_used,
                    generation_time=generation_time,
                )

            except Exception as e:
                logger.error(
                    f"Ошибка генерации (попытка {attempt + 1}/{max_attempts}): {e}"
                )
                if attempt < max_attempts - 1:
                    logger.info("Повторная попытка через 2 секунды...")
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Не удалось сгенерировать ответ: {e}") from e

    def generate_simple(self, prompt: str) -> str:
        """
        Простая генерация без системного промпта.

        :param prompt: промпт пользователя
        :return: текст ответа
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.exception("Ошибка простой генерации")
            raise RuntimeError(f"Ошибка генерации: {e}") from e
