import streamlit as st
import easyocr
from PIL import Image
import io
from io import StringIO
import random
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import requests
import time
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

st.set_page_config(page_title="DeepLearn", page_icon="📖")

# load in css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")

# question generator setup
question_model = "allenai/t5-small-squad2-question-generation"
tokenizer = T5Tokenizer.from_pretrained(question_model)
model = T5ForConditionalGeneration.from_pretrained(question_model)


@st.cache_resource
def generate_question(text, **generator_args):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output


# answer generator setup
answer_model = "consciousAI/question-answering-roberta-base-s"

with st.sidebar:
    image = Image.open("image/logo.png")
    st.sidebar.image(image)

    choices = ["Image", "Text"]
    choice = st.sidebar.selectbox(
        "Source", choices, help="You can upload an image or text file")

# init states
if "summary_button_clicked" not in st.session_state:
    st.session_state.summary_button_clicked = False
if "quiz_button_clicked" not in st.session_state:
    st.session_state.quiz_button_clicked = False
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "current_question_temp" not in st.session_state:
    st.session_state.current_question_temp = -1
if "new_source_summary" not in st.session_state:
    st.session_state.new_source_summary = False
if "new_source_quiz" not in st.session_state:
    st.session_state.new_source_quiz = False
if "guess" not in st.session_state:
    st.session_state.guess = ""
if "questions_answers" not in st.session_state:
    st.session_state.questions_answers = None
if "mock" not in st.session_state:
    st.session_state.mock = True
if "start" not in st.session_state:
    st.session_state.start = True

# =================================================== MOCK DATA START
mock_summary = '"Robot" comes from the Czech word robota, meaning"drudgery;" and first appeared in the 1921 play RUR. More than a million industrial robots are now in use, nearly half of them in Japan. Leonardo da Vinci drew up plans for an armored humanoid machine in 1495.'
mock_questions_answers = []
mock_questions_answers.append(("What is the name of the robots that scout for roadside bombs?", "Talon bots"))
mock_questions_answers.append(("What is the name of the robots that have logged 10.5 miles across the Red Planet", "Spirit and Opportunity"))
mock_questions_answers.append(("What was the name of the first robot built by Leonardo da Vinci?", "armored humanoid machine"))
mock_questions_answers.append(("How many industrial robots are now in use?", "More than a million"))
mock_questions_answers.append(("What is the Czech word for robota?", "drudgery"))
# =================================================== MOCK DATA END

def summary_button_callback():
    st.session_state.summary_button_clicked = True


def quiz_button_callback():
    st.session_state.quiz_button_clicked = True


def display_buttons():
    # render summarize / quiz buttons
    col1, col2 = st.columns(2)
    with col1:
        summary_button = st.button(
            ":robot_face: Summarize :scissors:", on_click=summary_button_callback, key="summary_button", use_container_width=True
        )
    with col2:
        quiz_button = st.button(
            ":robot_face: Quiz me! :pencil2:", on_click=quiz_button_callback, key="quiz_button", use_container_width=True
        )


def update_sidebar():
    st.sidebar.write("Original image")
    st.sidebar.image(st.session_state.image)
    with st.expander("See original text", expanded=True):
        st.write(easy_reading_text(st.session_state.extracted_text))


def easy_reading_text(text):
    words = text.split(' ')
    bold_words = []
    for word in words:
        if len(word) < 2:
            bold_words.append("**" + word + "**")
            continue
        num = int(len(word) / 2)
        bold_words.append("**" + word[:num] + "**" + word[num:])
    return ' '.join(bold_words)


tab1, tab2, tab3 = st.tabs(
    [":open_book: Main", ":student: Student", ":male-technologist: Professor"])

with tab1:
    if st.session_state.start:
        st.header(":arrow_left: Upload a file to begin :sparkles:")
        st.session_state.start = False
    if choice == "Image":
        uploaded_file = st.sidebar.file_uploader(
            "", type=['png'], help="Upload a file to begin!")

        if not st.session_state.summary_button_clicked and not st.session_state.quiz_button_clicked:
            image = None
            # get text from image using easyocr
            if uploaded_file is not None and not st.session_state.new_source_summary and not st.session_state.new_source_quiz:
                with st.spinner("Extracting text..."):
                    image = Image.open(uploaded_file)
                    st.sidebar.write("Original image")
                    st.sidebar.image(image)
                    st.session_state.image = image
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG',
                               subsampling=0, quality=100)
                    img_byte_arr = img_byte_arr.getvalue()
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext(img_byte_arr)

                    # prepare extracted text
                    text = [x[1] for x in result]
                    st.session_state.sentence_list = text
                    extracted_text_header = st.header(
                        "Extracted Text :open_book:")
                    st.session_state.extracted_text = ' '.join(text)
                    st.write(easy_reading_text(
                        st.session_state.extracted_text))
                    display_buttons()
                    st.session_state.new_source_summary = True
                    st.session_state.new_source_quiz = True

        # summary option
        if st.session_state.summary_button_clicked:
            update_sidebar()
            display_buttons()
            if st.session_state.mock:
                with st.spinner("Summarizing text..."):
                    time.sleep(2)
                    st.session_state.summary = mock_summary
                    st.header("Summary :pencil2:")
                    st.write(easy_reading_text(st.session_state.summary))
            else: 
                with st.spinner("Summarizing text..."):
                    if st.session_state.new_source_summary:
                        summarizer = pipeline(
                            "summarization", model='facebook/bart-large-cnn')
                        summary_raw = summarizer(st.session_state.extracted_text,
                                                max_length=200, min_length=30, do_sample=False)
                        st.session_state.summary = summary_raw[0]["summary_text"]
                        st.session_state.new_source_summary = False
                    st.header("Summary :pencil2:")
                    st.write(easy_reading_text(st.session_state.summary))

        # quiz option
        if st.session_state.quiz_button_clicked:
            update_sidebar()
            display_buttons()
            if st.session_state.new_source_quiz and st.session_state.questions_answers == None:
                if st.session_state.mock:
                    with st.spinner("Generating questions..."):
                        time.sleep(2)
                        st.session_state.questions_answers = mock_questions_answers
                        st.session_state.new_source_quiz = False
                else: 
                    with st.spinner("Generating questions..."):
                        if st.session_state.new_source_quiz:
                            paragraph = ""
                            paragraphs = []
                            for sentence in st.session_state.sentence_list:
                                paragraph += " " + sentence
                                if len(paragraph) > 200:
                                    paragraphs.append(paragraph)
                                    paragraph = ""
                            num_questions = min(len(paragraphs), 5)
                            questions_answers = []
                            question_answerer = pipeline(
                                "question-answering", model=answer_model)
                            for paragraph in paragraphs[:num_questions]:
                                question_list = generate_question(paragraph)
                                question = question_list[0]
                                answer = question_answerer(
                                    question=question, context=paragraph)
                                questions_answers.append(
                                    (question, answer['answer']))
                            st.session_state.questions_answers = questions_answers
                            st.session_state.new_source_quiz = False

                # while st.session_state.current_question == st.session_state.current_question_temp:
                #     st.session_state.current_question = (random.randint(0, 4))
                # question = st.session_state.questions_answers[st.session_state.current_question][0]
                # answer = st.session_state.questions_answers[st.session_state.current_question][1]
                # st.session_state.current_question_temp = st.session_state.current_question

                with st.spinner("Pop quiz time! :sparkles: :100:"):
                    for i in range(5):
                        st.session_state.current_question = i
                        question = st.session_state.questions_answers[st.session_state.current_question][0]
                        answer = st.session_state.questions_answers[st.session_state.current_question][1]

                        question_element = st.markdown(
                            f'<div class="blockquote-wrapper"><div class="blockquote"><h1><span style="color:#1e1e1e;">{question}</span></h1><h4>&mdash; - DeepLearn</em></h4></div></div>',
                            unsafe_allow_html=True,
                        )

                        answer_element = None
                        with st.spinner("Revealing answer in 5 seconds..."):
                            time.sleep(5)
                            answer_element = st.markdown(
                                f"<div class='answer'><span style='font-weight: bold; color:#6d7284;'>Answer:</span><br><br>{answer}</div>",
                                unsafe_allow_html=True,
                            )
                        time.sleep(2)
                        question_element.empty()
                        answer_element.empty()
                        st.balloons()

    elif choice == "Text":
        uploaded_file = st.sidebar.file_uploader(
            "", type=['txt'])

        if not st.session_state.summary_button_clicked and not st.session_state.quiz_button_clicked:
            if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                st.write(bytes_data)

                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                st.write(stringio)

                # To read file as string:
                text = stringio.read()

    st.session_state.summary_button_clicked = False
    st.session_state.quiz_button_clicked = False

with tab2:
    # Mock data
    quiz_scores = [75, 90, 85, 78, 92, 88, 96, 81, 95, 89, 82, 91, 84, 77, 99]

    quiz_names = [
        "Introduction to AI",
        "Machine Learning Basics",
        "Deep Learning",
        "Neural Networks",
        "Computer Vision",
        "Natural Language Processing",
        "Reinforcement Learning",
        "Generative Models",
        "AI Ethics",
        "Robotics",
        "AI in Healthcare",
        "AI in Finance",
        "AI in Agriculture",
        "AI in Manufacturing",
        "AI in Transportation",
    ]

    quiz_data = pd.DataFrame({"Quiz": quiz_names, "Score": quiz_scores,
                             "Quiz Number": list(range(1, len(quiz_scores) + 1))})

    # All quizzes chart
    st.header("Quiz Scores by Topic :sparkles:")
    st.markdown(
        "A visualization of your strengths and weaknesses")

    fig = px.line_polar(quiz_data, r='Score', theta='Quiz', line_close=True, range_r=[
                        (min(quiz_scores)-5), max(quiz_scores)])
    # Change the range_r numbers color to black
    fig.update_polars(angularaxis=dict(showline=True, linecolor="black",
                      linewidth=2, gridcolor="white", gridwidth=1, tickfont=dict(color="black")))

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Summary chart
    average_score = quiz_data["Score"].mean()
    min_score = quiz_data["Score"].min()
    max_score = quiz_data["Score"].max()

    st.header("Summary Statistics :sparkles:")
    st.markdown("A quick summary of your quiz scores")
    summary_data = pd.DataFrame({
        "Statistic": ["Average Score", "Lowest Score", "Highest Score"],
        "Score": [average_score, min_score, max_score],
    })
    fig = px.bar(summary_data, x="Statistic", y="Score",
                 title="", color_discrete_sequence=["#9EE6CF"])
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    
    # Fit a linear regression model
    X = quiz_data["Quiz Number"].values.reshape(-1, 1)
    y = quiz_data["Score"].values.reshape(-1, 1)
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    y_pred = regression_model.predict(X)

    # Quiz scores with linear regression chart
    st.header("Linear Regression of Quiz Scores :sparkles:")
    st.markdown("An indication of how your scores a trending")

    fig = px.scatter(quiz_data, x="Quiz Number", y="Score",
                     text="Quiz", color_discrete_sequence=["#9EE6CF"])
    fig.add_trace(px.line(
        x=quiz_data["Quiz Number"], y=y_pred.reshape(-1), markers=False).data[0])
    fig.update_xaxes(title_text="Quiz Number")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme=None, use_container_width=True)

    

    # Predicts next score, creates dataframe for previous score and predicted
    next_quiz_number = len(quiz_scores) + 1
    next_quiz_score = regression_model.predict(
        np.array([[next_quiz_number]]))[0][0]

    prev_and_projected_data = pd.DataFrame({
        "Type": ["Previous Quiz Score", "Projected Score for Next Quiz"],
        "Score": [quiz_scores[-1], next_quiz_score],
    })

    # Chart for previous and predicted
    st.header("Previous Quiz Score vs Projected Score for Next Quiz")
    st.markdown("Expected score on your next quiz based on recent performance")
    fig = px.bar(prev_and_projected_data, x="Type", y="Score",
                 title="", color_discrete_sequence=["#9EE6CF"])
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with tab3:

    # Number of students
    num_students = 10

    # Mock data for all students
    all_students_scores = [
        [75, 90, 85, 78, 92, 88, 96, 81, 95, 89, 82, 91, 84, 99, 73],
        [68, 84, 76, 82, 89, 78, 91, 74, 92, 86, 81, 87, 79, 97, 79],
        [72, 88, 81, 77, 95, 83, 94, 80, 93, 84, 79, 90, 82, 98, 85],
        [65, 81, 74, 71, 88, 75, 87, 70, 86, 80, 75, 84, 78, 95, 69],
        [78, 92, 86, 83, 97, 89, 99, 85, 98, 90, 84, 93, 87, 100, 75],
        [71, 85, 80, 76, 90, 82, 93, 78, 91, 83, 77, 88, 81, 96, 82],
        [74, 88, 83, 80, 93, 87, 96, 82, 95, 89, 83, 92, 86, 99, 74],
        [67, 80, 75, 72, 86, 78, 89, 71, 88, 82, 76, 85, 79, 94, 83],
        [70, 84, 79, 75, 89, 81, 92, 76, 91, 85, 80, 87, 82, 97, 80],
        [73, 87, 82, 78, 92, 85, 95, 79, 94, 88, 83, 90, 85, 98, 77],
    ]

    all_students_data = []

    for i in range(num_students):
        student_data = pd.DataFrame({"Quiz": quiz_names,
                                     "Score": all_students_scores[i],
                                     "Quiz Number": list(range(1, len(quiz_scores) + 1)),
                                     "Student ID": [i+1]*len(quiz_scores)})
        all_students_data.append(student_data)

    all_students_data_combined = pd.concat(
        all_students_data, ignore_index=True)

    # Get the latest quiz scores for each student
    latest_quiz_scores = [scores[-1] for scores in all_students_scores]

    # Linear Regression of Quiz Scores for All Students
    st.header("Linear Regression of Quiz Scores for All Students :sparkles:")
    st.markdown("Latest score trends for your students")
    fig = go.Figure()

    combined_data = pd.concat(all_students_data, ignore_index=True)

    X = combined_data["Quiz Number"].values.reshape(-1, 1)
    y = combined_data["Score"].values.reshape(-1, 1)
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    y_pred = regression_model.predict(X)

    for student_id, student_data in enumerate(all_students_data):
        fig.add_trace(go.Scatter(
            x=student_data["Quiz Number"], y=student_data["Score"], mode='markers', name=f"Student {student_id + 1}"))

    fig.add_trace(go.Scatter(x=combined_data["Quiz Number"], y=y_pred.reshape(
        -1), mode='lines', name="All Students Regression"))

    fig.update_xaxes(title_text="Quiz Number")
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, theme=None, use_container_width=True)

    # Create the table for performance on the last quiz
    fig = go.Figure()

    st.header("Class Quiz Scores :sparkles:")
    st.markdown("Tracks recent student performance")


    header_labels = ['Student', 'Test Name', 'Performance']
    for i, label in enumerate(header_labels):
        fig.add_shape(type='rect', xref='x', yref='y', x0=i - 0.5, x1=i + 0.5, y0=len(latest_quiz_scores),
                      y1=len(latest_quiz_scores) + 1, fillcolor='rgba(0, 0, 0, 0.57)', line=dict(color='white'))
        fig.add_annotation(x=i, y=len(latest_quiz_scores) + 0.25,
                           text=label, font=dict(size=12, color='white'), showarrow=False)

    for row, score in enumerate(latest_quiz_scores):
        green_percentage = score / 100
        red_percentage = 1 - green_percentage

        # Add student information
        fig.add_shape(type='rect', xref='x', yref='y', x0=-0.5, x1=0.5,
                      y0=row, y1=row + 1, fillcolor='white', line=dict(color='white'))
        fig.add_annotation(
            x=0, y=row + 0.5, text=f"Student {row + 1}", font=dict(size=11, color='black'), showarrow=False)

        # Add test name
        fig.add_shape(type='rect', xref='x', yref='y', x0=0.5, x1=1.5,
                      y0=row, y1=row + 1, fillcolor='white', line=dict(color='white'))
        fig.add_annotation(
            x=1, y=row + 0.5, text=quiz_names[-1], font=dict(size=11, color='black'), showarrow=False)

        # Add performance
        fig.add_shape(type='rect', xref='x', yref='y', x0=1.5, x1=1.5 + green_percentage,
                      y0=row, y1=row + 1, fillcolor='rgba(118, 255, 162, 0.77)', line=dict(color='white'))
        fig.add_shape(type='rect', xref='x', yref='y', x0=1.5 + green_percentage, x1=2.5,
                      y0=row, y1=row + 1, fillcolor='rgba(255, 91, 52, 0.57)', line=dict(color='white'))
        fig.add_annotation(
            x=2, y=row + 0.5, text=f"{score}%", font=dict(size=11, color='black'), showarrow=False)

    fig.update_xaxes(showgrid=False, zeroline=False,
                     visible=False, range=[-0.5, 2.5])
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False,
                     range=[-0.5, len(latest_quiz_scores) + 0.5], autorange='reversed')
    fig.update_layout(title='Latest Quiz Performance', width=800, height=40 *
                      (len(latest_quiz_scores) + 1), margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
