# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW  # Теперь импортируем из torch.optim
import numpy as np
import random
from tqdm import tqdm
import re


# Генерация синтетического датасета для фитнес-тренера
def generate_fitness_dataset(num_samples=500):
    dataset = []

    # Шаблоны вопросов и ответов
    exercise_questions = [
        "Как правильно делать приседания?",
        "Техника выполнения становой тяги",
        "Как делать жим лежа без ошибок?",
        "Правильная форма для подтягиваний",
        "Как выполнять выпады?",
        "Техника отжиманий для начинающих",
        "Как правильно делать планку?",
        "Техника бега на длинные дистанции",
        "Как плавать кролем?",
        "Правильная техника прыжков со скакалкой"
    ]

    nutrition_questions = [
        "Сколько белка нужно в день?",
        "Что есть перед тренировкой?",
        "Что есть после тренировки?",
        "Как рассчитать калории для похудения?",
        "Лучшие источники полезных жиров",
        "Сколько воды нужно пить при тренировках?",
        "Как питаться для набора мышечной массы?",
        "Что такое гликемический индекс?",
        "Нужны ли спортивные добавки?",
        "Пример рациона на день"
    ]

    sleep_questions = [
        "Сколько нужно спать при активных тренировках?",
        "Как улучшить качество сна?",
        "Влияет ли сон на восстановление мышц?",
        "Какой режим сна оптимален?",
        "Что делать при бессоннице?",
        "Стоит ли спать днем после тренировки?",
        "Как сон влияет на метаболизм?",
        "Позы для сна при болях в спине",
        "Как восстановить режим сна?",
        "Вредно ли спать меньше 6 часов?"
    ]

    routine_questions = [
        "Как составить программу тренировок?",
        "Сколько раз в неделю тренироваться?",
        "Как совмещать кардио и силовые тренировки?",
        "Идеальный утренний распорядок",
        "Как восстановиться после тренировки?",
        "Нужны ли отдыхать между подходами?",
        "Сколько времени должна длиться тренировка?",
        "Как часто менять программу тренировок?",
        "Разминка перед тренировкой",
        "Заминка после тренировки"
    ]

    # Ответы
    exercise_answers = [
        "Держите спину прямой, ноги на ширине плеч. Опускайтесь до параллели с полом, колени не должны выходить за носки.",
        "Спина прямая, ноги на ширине плеч. Поднимайте вес усилием мышц спины и ног, а не рук. Не округляйте поясницу!",
        "Лягте на скамью, сведите лопатки. Опускайте гриф к середине груди, локти под углом 45 градусов.",
        "Возьмитесь за перекладину шире плеч. Подтягивайтесь усилием спины, а не рук. В верхней точке подбородок выше перекладины.",
        "Шагните вперед, опускайтесь до параллели бедра с полом. Колено не должно выходить за носок. Спина прямая.",
        "Руки на ширине плеч, тело образует прямую линию. Опускайтесь до угла 90 градусов в локтях.",
        "Упритесь на предплечья и носки. Тело должно быть параллельно полу. Напрягите пресс и ягодицы.",
        "Держите корпус прямо, руки согнуты под 90 градусов. Приземляйтесь на переднюю часть стопы. Дыхание ритмичное.",
        "Лицо в воде, делайте гребки попеременно руками, ноги работают как ножницы. Выдыхайте в воду.",
        "Держите локти близко к телу, вращайте скакалку запястьями. Приземляйтесь на носки."
    ]

    nutrition_answers = [
        "1.6-2.2 г на кг веса тела. При наборе массы - до 2.5 г/кг.",
        "За 1-2 часа до тренировки: сложные углеводы + белок (овсянка + яйца, гречка + курица).",
        "В течение 30 минут после: белок + быстрые углеводы (сывороточный протеин + банан, творог + мед).",
        "Ваш вес (кг) * 30 - 500 ккал для похудения. Минимум 1200 ккал для женщин, 1500 для мужчин.",
        "Авокадо, орехи, жирная рыба, оливковое масло, семена льна и чиа.",
        "30-40 мл на кг веса. При интенсивных тренировках +500-1000 мл.",
        "Профицит калорий + 2 г белка/кг + силовые тренировки. Увеличивайте калорийность постепенно.",
        "Показатель скорости усвоения углеводов. Выбирайте продукты с низким ГИ (овсянка, гречка, овощи).",
        "Протеин и креатин имеют доказанную эффективность. Остальные - по желанию и целям.",
        "Завтрак: овсянка + яйца. Обед: гречка + курица + овощи. Ужин: рыба + салат. Перекусы: творог, фрукты, орехи."
    ]

    sleep_answers = [
        "7-9 часов. При интенсивных тренировках - не менее 8 часов.",
        "За 1 час до сна избегайте экранов. Проветривайте комнату. Температура 18-20°C. Расслабляющие ритуалы.",
        "Да, 70% гормона роста вырабатывается в фазе глубокого сна. Это критично для восстановления мышц.",
        "Ложитесь до 23:00, вставайте в 6-8 утра. Старайтесь соблюдать график даже в выходные.",
        "Не пользуйтесь гаджетами за час до сна. Попробуйте медитацию. Если не спится 20 минут - встаньте и почитайте.",
        "Да, 20-30 минут сна улучшают восстановление. Но не спите дольше, чтобы не нарушить ночной сон.",
        "Недосып снижает чувствительность к инсулину, повышает кортизол и тягу к сладкому.",
        "На боку с подушкой между коленями. Или на спине с подушкой под коленями.",
        "Каждый день вставайте на 15 минут раньше. Утром выходите на свет. Вечером приглушайте свет.",
        "Да, хронический недосып повышает риски ожирения, диабета и сердечно-сосудистых заболеваний."
    ]

    routine_answers = [
        "3 силовые тренировки в неделю на все тело или сплит (ноги/спина/грудь). Добавьте 2 кардио сессии.",
        "3-5 раз в неделю. Начинающим - 3 раза с отдыхом между днями.",
        "После силовой тренировки или в отдельные дни. Либо утром кардио, вечером силовая.",
        "Пробуждение, стакан воды, зарядка 10 мин, контрастный душ, полезный завтрак, планирование дня.",
        "Растяжка, массаж, сауна, сон 8 часов, активное восстановление (легкая ходьба, плавание).",
        "Да, 60-90 секунд между подходами на силу, 30-45 сек на выносливость.",
        "Силовая тренировка 45-60 минут, кардио 30-45 минут. Не считая разминку и заминку.",
        "Каждые 6-8 недель меняйте упражнения, порядок, количество подходов/повторений.",
        "5-10 минут динамической растяжки и легкого кардио (бег на месте, прыжки).",
        "5-10 минут статической растяжки работавших мышц. Восстановление дыхания."
    ]

    # Сборка датасета
    for i in range(num_samples):
        category = random.choice(['exercise', 'nutrition', 'sleep', 'routine'])

        if category == 'exercise':
            q = random.choice(exercise_questions)
            a = random.choice(exercise_answers)
        elif category == 'nutrition':
            q = random.choice(nutrition_questions)
            a = random.choice(nutrition_answers)
        elif category == 'sleep':
            q = random.choice(sleep_questions)
            a = random.choice(sleep_answers)
        else:
            q = random.choice(routine_questions)
            a = random.choice(routine_answers)

        # Добавляем вариативности
        patterns = [
            f"Вопрос: {q}\nОтвет: {a}",
            f"Q: {q}\nA: {a}",
            f"Пользователь: {q}\nТренер: {a}",
            f"{q}\n{a}"
        ]

        dataset.append(random.choice(patterns))

    return dataset


# Класс для обработки датасета
class FitnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for text in data:
            encodings = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            self.input_ids.append(encodings['input_ids'])
            self.attn_masks.append(encodings['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].squeeze(),
            'attention_mask': self.attn_masks[idx].squeeze()
        }


# Параметры обучения
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
MAX_LENGTH = 256
MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"

# Инициализация модели и токенизатора
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# Генерация и подготовка датасета
print("Генерация датасета...")
fitness_data = generate_fitness_dataset(2000)  # 2000 примеров
train_dataset = FitnessDataset(fitness_data, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Оптимизатор (исправленная строка)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Начало обучения на {device}...")

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} завершен. Средний loss: {avg_loss:.4f}")

# Сохранение модели
model.save_pretrained("fitness_trainer_model")
tokenizer.save_pretrained("fitness_trainer_tokenizer")
print("Модель сохранена!")


# Функция для генерации ответа
def generate_response(question, model, tokenizer):
    prompt = f"Вопрос: {question}\nОтвет:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True
    ).to(device)

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=MAX_LENGTH,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Ответ:")[-1].strip()


# Интерактивный режим
def interactive_chat():
    print("\nФитнес-бот готов к общению! (введите 'выход' для завершения)")

    while True:
        user_input = input("Вы: ")

        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("До свидания! Желаю успехов в тренировках!")
            break

        response = generate_response(user_input, model, tokenizer)
        print(f"Тренер: {response}")


# Запуск чата
interactive_chat()