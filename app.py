import streamlit as st
import joblib
import pandas as pd

model = joblib.load("titanic_model.pkl")
feature_names = joblib.load("feature_names.pkl")
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

st.header("Passenger Information")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

embarked = st.selectbox(
    "Embarked",
    ["S", "C", "Q"]
)

cabin_known = st.selectbox(
    "Cabin Information Available?",
    [0, 1]
)

title = st.selectbox(
    "Title",
    [
        "Mr", "Mrs", "Miss", "Master",
        "Dr", "Rev", "Col", "Major",
        "Mlle", "Countess", "Ms", "Lady",
        "Jonkheer", "Don", "Mme", "Sir"
    ]
)

deck = st.selectbox(
    "Deck",
    ["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"]
)
# Feature Engineering

family_size = sibsp + parch + 1

is_alone = 1 if family_size == 1 else 0

# Age Group
if age < 12:
    age_group = "Child"
elif age < 20:
    age_group = "Teen"
elif age < 60:
    age_group = "Adult"
else:
    age_group = "Senior"

# Fare per person
fare_per_person = fare / family_size
passenger = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "CabinKnown": [cabin_known],
    "FarePerPerson": [fare_per_person],
    "Sex": [sex],
    "Embarked": [embarked],
    "Title": [title],
    "Deck": [deck],
    "AgeGroup": [age_group]
})
passenger = pd.get_dummies(
    passenger,
    columns=["Sex", "Embarked", "Title", "Deck", "AgeGroup"],
    drop_first=True
)
passenger = passenger.reindex(
    columns=feature_names,
    fill_value=0
)
if st.button("Predict Survival"):

    prediction = model.predict(passenger)
    probability = model.predict_proba(passenger)

    if prediction[0] == 1:
        st.success("The passenger is predicted to survive!")
    else:
        st.error("The passenger is predicted not to survive.")

    st.write(
        f"Survival probability: {probability[0][1] * 100:.2f}%"
    )