import os
import json
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler


class FitnessTrainerBot:
    def __init__(self, openrouter_api_key: str):
        self.openrouter_api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/IvanZhutyaev/FitnessTrackingApp",
            "X-Title": "ChatBotKey"
        }

    async def get_ai_response(self, prompt: str) -> str:
        request_data = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Вы — квалифицированный фитнес-тренер. Ваша задача — давать чёткие, профессиональные "
                        "и обоснованные ответы на вопросы, касающиеся сна, питания, физических тренировок, "
                        "восстановления, режима дня, образа жизни и мотивации. Вы не отвечаете на вопросы, "
                        "не связанные с данной тематикой. В таких случаях следует строго и сдержанно сообщить, "
                        "что ваша компетенция ограничена вопросами, касающимися фитнеса, здоровья и тренировочного процесса."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=request_data
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            return content.strip() if content else "Пустой ответ от модели."

        except Exception as e:
            return f"Ошибка: {str(e)}"


class TelegramBot:
    def __init__(self, token: str, fitness_bot: FitnessTrainerBot):
        self.token = token
        self.fitness_bot = fitness_bot
        self.app = Application.builder().token(self.token).build()

        # Регистрация обработчиков
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, context: CallbackContext) -> None:
        """Отправляет приветственное сообщение при команде /start"""
        welcome_text = (
            "👋 Привет! Я ваш виртуальный фитнес-тренер.\n\n"
            "Задавайте любые вопросы о тренировках, питании и здоровом образе жизни!\n\n"
            "Примеры вопросов:\n"
            "- Как составить программу тренировок для начинающих?\n"
            "- Какое питание рекомендуется после силовой тренировки?\n"
            "- Как правильно выполнять приседания?\n"
            "- Сколько раз в неделю нужно тренироваться?"
        )

        await update.message.reply_text(welcome_text)

    async def help_command(self, update: Update, context: CallbackContext) -> None:
        """Отправляет сообщение с помощью при команде /help"""
        help_text = (
                "🤖 Я могу помочь вам с вопросами о:\n"
                "- Тренировках и упражнениях\n"
                "- Питании и диетах\n"
        "- Восстановлении после тренировок\n"
        "- Режиме дня и здоровом образе жизни\n"
        "- Мотивации и постановке целей\n\n"
        "Просто задайте ваш вопрос в чат!"
        )
        await update.message.reply_text(help_text)

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        """Обрабатывает текстовые сообщения пользователя"""
        user_message = update.message.text

        # Отправляем статус "печатает"
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Получаем ответ от ИИ
        ai_response = await self.fitness_bot.get_ai_response(user_message)

        # Отправляем ответ пользователю
        await update.message.reply_text(ai_response)

    def run(self):
        """Запускает бота"""
        self.app.run_polling()


if __name__ == "__main__":
    # Конфигурация
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Инициализация и запуск бота
    fitness_bot = FitnessTrainerBot(OPENROUTER_API_KEY)
    telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN, fitness_bot)
    telegram_bot.run()