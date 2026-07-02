from flask import Flask, render_template,request
import pandas as pd
import pickle

app=Flask(__name__)

#load model
model=pickle.load(open("sysmptom_disease.pkl","rb"))

#Load Dataset
df=pd.read_csv("Diseases_and_Symptoms_dataset.csv")
desc1=pd.read_csv("description.csv")
desc2=pd.read_csv("medications.csv")
desc3=pd.read_csv("precautions.csv")
desc4=pd.read_csv("workout.csv")
desc5=pd.read_csv("diets.csv")

# To access columns
symptoms=list(df.columns[1:])

@app.route("/")
def home():
    return render_template("index.html",symptoms=symptoms)
    
@app.route("/predict",methods=["POST"])
def predict():
    selected_symptoms=request.form.getlist("symptoms")

    input_data=[0]*len(symptoms)

    for symptom in selected_symptoms:
        if symptom in symptoms:
            index=symptoms.index(symptom)
            input_data[index]=1
    disease=model.predict([input_data])[0]
    

    try:
        accuracy=round(max(model.predict_proba([input_data])[0])*100,2)
    except:
        accuracy="N/A"

    desc=desc1[
            desc1["Disease"].str.lower()== disease.lower()]


    med=desc2[
            desc2["Disease"].str.lower()== disease.lower()]


    pre=desc3[
            desc3["Disease"].str.lower()== disease.lower()]


    work=desc4[
            desc4["Disease"].str.lower()== disease.lower()]


    diet_plan=desc5[
            desc5["Disease"].str.lower()== disease.lower()]

    return render_template("result.html",disease=disease, selected_symptoms=selected_symptoms,accuracy=accuracy,desc=desc,med=med,pre=pre,diet_plan=diet_plan,work=work
    )                           

if __name__== "__main__":
    app.run(debug=True)






 