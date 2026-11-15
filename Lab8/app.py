import gradio as gr
from sql_generator import generate_sql, MODEL

SCHEMA_SHOP = """CREATE TABLE Users (
    Id INT PRIMARY KEY,
    FullName VARCHAR(100),
    Email VARCHAR(100),
    CreatedAt DATETIME
);

CREATE TABLE Orders (
    Id INT PRIMARY KEY,
    UserId INT FOREIGN KEY REFERENCES Users(Id),
    TotalAmount DECIMAL(10, 2),
    Status VARCHAR(20),
    OrderDate DATETIME
);"""

SCHEMA_LIBRARY = """CREATE TABLE Authors (
    Id INT PRIMARY KEY,
    Name VARCHAR(100),
    Country VARCHAR(50)
);

CREATE TABLE Books (
    Id INT PRIMARY KEY,
    Title VARCHAR(200),
    AuthorId INT REFERENCES Authors(Id),
    PublishedYear INT,
    Genre VARCHAR(50)
);

CREATE TABLE Readers (
    Id INT PRIMARY KEY,
    FullName VARCHAR(100),
    RegisteredAt DATE
);

CREATE TABLE Loans (
    Id INT PRIMARY KEY,
    BookId INT REFERENCES Books(Id),
    ReaderId INT REFERENCES Readers(Id),
    LoanDate DATE,
    ReturnDate DATE NULL
);"""

SCHEMA_UNIVERSITY = """CREATE TABLE Students (
    Id INT PRIMARY KEY,
    FullName VARCHAR(100),
    GroupName VARCHAR(20),
    EnrollmentYear INT
);

CREATE TABLE Courses (
    Id INT PRIMARY KEY,
    Title VARCHAR(100),
    Credits INT,
    Teacher VARCHAR(100)
);

CREATE TABLE Grades (
    StudentId INT REFERENCES Students(Id),
    CourseId INT REFERENCES Courses(Id),
    Score INT,
    ExamDate DATE,
    PRIMARY KEY (StudentId, CourseId)
);"""

SCHEMA_HR = """CREATE TABLE Departments (
    Id INT PRIMARY KEY,
    Name VARCHAR(100),
    Location VARCHAR(100)
);

CREATE TABLE Employees (
    Id INT PRIMARY KEY,
    FullName VARCHAR(100),
    DepartmentId INT REFERENCES Departments(Id),
    ManagerId INT REFERENCES Employees(Id),
    Salary DECIMAL(10, 2),
    HireDate DATE
);"""

DEFAULT_SCHEMA = SCHEMA_SHOP

DEFAULT_QUERY = "Знайди суму всіх сплачених замовлень (Status = 'Paid') за останній місяць для кожного користувача"

EXAMPLES = [
    ["PostgreSQL", SCHEMA_SHOP, "Виведи 5 користувачів з найбільшою кількістю замовлень"],
    ["MySQL", SCHEMA_SHOP, "Знайди користувачів, які зареєструвалися, але не зробили жодного замовлення"],

    ["PostgreSQL", SCHEMA_LIBRARY, "Покажи всі книги, які зараз на руках у читачів, з іменем читача"],
    ["SQLite", SCHEMA_LIBRARY, "Знайди читачів, які тримають книгу понад 30 днів і досі не повернули"],
    ["T-SQL (MS SQL)", SCHEMA_LIBRARY, "Для кожного жанру виведи найпопулярнішу книгу за кількістю видач"],

    ["PostgreSQL", SCHEMA_UNIVERSITY, "Середній бал по кожній групі, відсортований за спаданням"],
    ["MySQL", SCHEMA_UNIVERSITY, "Студенти, у яких є хоча б одна оцінка нижче 60"],
    ["PostgreSQL", SCHEMA_UNIVERSITY, "Для кожного курсу виведи студента з найвищим балом"],

    ["T-SQL (MS SQL)", SCHEMA_HR, "Співробітники, які заробляють більше за свого керівника"],
    ["PostgreSQL", SCHEMA_HR, "Відділи, де середня зарплата вища за середню по компанії"],
    ["MySQL", SCHEMA_HR, "Виведи ієрархію підлеглих для співробітника з Id = 1 (всі рівні)"],
]

with gr.Blocks(title="NL to SQL Generator") as demo:
    gr.Markdown("# Конвертатор текстових запитів у SQL")
    gr.Markdown(
        "Сервіс для автоматичної генерації SQL-запитів із текстових вимог природною мовою "
        f"на основі наданої схеми БД за допомогою моделі `{MODEL}`."
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenAI API Key (Введіть вручну або він буде взятий із файлу .env, якщо такий існує)",
            placeholder="sk-proj-...",
            type="password",
        )
        dialect_input = gr.Dropdown(
            label="СУБД / Діалект",
            choices=["PostgreSQL", "T-SQL (MS SQL)", "MySQL", "SQLite"],
            value="PostgreSQL",
        )


    schema_input = gr.Code(
        label="Схема бази даних (DDL)",
        value=DEFAULT_SCHEMA,
        language="sql",
        lines=8,
    )


    query_input = gr.Textbox(
        label="Опис запиту природною мовою",
        value=DEFAULT_QUERY,
        lines=3,
        render=False
    )
    gr.Examples(
        examples=EXAMPLES,
        inputs=[dialect_input, schema_input, query_input],
        label="Приклади (клікніть, щоб підставити)",
        examples_per_page=2
    )

    query_input.render()


    submit_btn = gr.Button("Згенерувати SQL", variant="primary")

    sql_output = gr.Code(
        label="Згенерований SQL-код",
        language="sql",
        interactive=False,
    )

    submit_btn.click(
        fn=generate_sql,
        inputs=[api_key_input, dialect_input, schema_input, query_input],
        outputs=sql_output,
    )

if __name__ == "__main__":
    demo.launch()