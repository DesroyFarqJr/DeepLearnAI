# DeepLearn

Hosted at: https://hackmiami.streamlit.app/

The demo version is using data that's been generated by the models we're using. We've provided 1 example of the workflow.
We can load, extract, and process multiple images in the local version, but we could not find a service that could host all the models that we are using.

You should be able to
1. Extract text from uploaded image using **EasyOCR**
2. Summarize - isolates the important points in the extracted text using the **facebook/bart-large-cnn** model
3. Generate and view a quiz containing questions generated from the **allenai/t5-small-squad2-question-generation** model and answers generated by the **consciousAI/question-answering-roberta-base-s** model

Enjoy!
