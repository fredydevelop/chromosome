#Importing the dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import streamlit as st
import base64
import pickle as pk




#configuring the page setup
st.set_page_config(page_title='Anomalies in Chromosome',layout='centered')

#selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],menu_icon="house",default_index=0)
with st.sidebar:
    st.title("Home Page")
    selection=st.radio("select your option",options=["Predict for a Single-Patient", "Predict for Multi-Patient"])


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


#single prediction function
def chromosomeAnom(givendata):
    
    loaded_model=pk.load(open("chromosomeAnomaliesSinglePredict.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction==1 or prediction=="1":
      return "There is anomalies present in chromosomes"
    else:
      return "No anomalies present"
    
 
#main function handling the input
def main():
    st.header("Chromosome Anomalies Prediction System")
    
    #getting user input
    
    age = st.slider('Patient age', 0, 80, 18, key="ageslide")
    st.write("Patient is", age, 'years old')

    option1 = st.selectbox('Gender',("",'Male' ,'Female'),key="gender")
    if (option1=='Male'):
        Gender=1
    else:
        Gender=0
    
    option2 = st.selectbox('Ethnicity ', ("",'Hispanic','Asian','White','Black','Other'),key="Ethnic")
    if (option2=='Black'):
        Ethnicity=1
    elif (option2=="White"):
        Ethnicity=4
    elif (option2=="Asian"):
        Ethnicity=0
    elif (option2=="Other"):
        Ethnicity=3
    else:
        Ethnicity=2

    
    #RegisteredPhoneNumber=st.text_input("Has number been registered ?")


    option4 = st.selectbox("Patient medical history",("",'Asthma','Migraines','Depression','Hypertension','Thyroid disease','High blood pressure','Chronic obstructive pulmonary disease (COPD)','Other','Diabetes','None','Arthritis','Anxiety','Cancer','Heart disease'),key="medical")
    if (option4=='Asthma'):
        medical_history=2
    elif (option4=="Migraines"):
        medical_history=10
    elif (option4=="Depression"):
        medical_history=5
    elif (option4=="Hypertension"):
        medical_history=9
    elif (option4=="Thyroid disease"):
        medical_history=13
    elif (option4=="High blood pressure"):
        medical_history=8
    elif (option4=="Chronic obstructive pulmonary disease (COPD)"):
        medical_history=4
    elif (option4=="Other"):
        medical_history=12
    elif (option4=="Diabetes"):
        medical_history=6
    elif (option4=="None"):
        medical_history=11
    elif (option4=="Arthritis"):
        medical_history=1
    elif (option4=="Anxiety"):
        medical_history=0
    elif (option4=="Cancer"):
        medical_history=3
    else:
        medical_history=7
        
    #ActiveFor3monthsAndAbove=st.text_input("Been active for three months and above ?")
    # code for Prediction
    

    option5 = st.selectbox("family medical history",("",'Cancer','Stroke','None','Other','hypertension','Heart disease','diabetes','asthma'),key="family_medical")
    if (option5=='Cancer'):
        family_medical_history=0
    elif (option5=="Stroke"):
        family_medical_history=4
    elif (option5=="None"):
        family_medical_history=2
    elif (option5=="Other"):
        family_medical_history=3
    elif (option5=="hypertension"):
        family_medical_history=7
    elif (option5=="Heart disease"):
        family_medical_history=1
    elif (option5=="diabetes"):
        family_medical_history=6
    else:
        family_medical_history=5


    

    option13 = st.selectbox("telomere length: ",("",'low','Normal','high'),key="telomere")
    if (option13=='low'):
        telomere_length=2
    elif (option13=="Normal"):
        telomere_length=0
    else:
        telomere_length=1
        
    
    

    option14 = st.selectbox("medications taken ",("",'Cisplatin','Etoposide','Paclitaxel','Chlorambucil','Doxorubicin','Methotrexate','Bleomycin','None','Busulfan','Vincristine','Cyclophosphamide','Other'),key="medic")
    if (option14=='Cisplatin'):
        medications_taken=3
    elif (option14=="Etoposide"):
        medications_taken=6
    elif (option14=="Paclitaxel"):
        medications_taken=10
    elif (option14=="Chlorambucil"):
        medications_taken=2
    elif (option14=="Doxorubicin"):
        medications_taken=5
    elif (option14=="Methotrexate"):
        medications_taken=7
    elif (option14=="Bleomycin"):
        medications_taken=0
    elif (option14=="None"):
        medications_taken=8
    elif (option14=="Busulfan"):
        medications_taken=1
    elif (option14=="Vincristine"):
        medications_taken=11
    elif (option14=="Cyclophosphamide"):
        medications_taken=4
    else:
        medications_taken=9


    option15 = st.selectbox("lifestyle habits",("",'Poor diet','hard drinker','Current smoker','Lack of sleep','Non-smoker','Other'),key="habits")
    if (option15=='Poor diet'):
        lifestyle_habits=4
    elif (option15=='hard drinker'):
        lifestyle_habits=5
    elif (option15=='Current smoker'):
        lifestyle_habits=0
    elif (option15=='Lack of sleep'):
        lifestyle_habits=2 
    elif (option15=="Non-smoker"):
        lifestyle_habits=3
    else:
        lifestyle_habits=1




    option16 = st.selectbox ("environmental exposure",("",'None','Other','Air pollution','Second-hand smoke','Pesticides','Occupational exposure','Radiation'),key="environmental_expo")
    if (option16=='None'):
        environmental_exposure=1
    elif (option16== 'Other'):
        environmental_exposure=3
    elif (option16== 'Air pollution'):
        environmental_exposure=0
    elif (option16== 'Second-hand smoke'):
        environmental_exposure=6
    elif (option16== 'Pesticides'):
        environmental_exposure=4
    elif (option16== 'Occupational exposure'):
        environmental_exposure=2
    else:
        environmental_exposure=5

    option3 = st.selectbox("occupation",("",'Other','Teacher','Nurse','Office worker','Construction worker','Engineer'),key="occupationwork")
    if option3 == "Other":
        occupation=4 
    elif (option3=="Teacher"):
        occupation=5 
    elif (option3=="Nurse"):
        occupation=2 
    elif (option3=="Office worker"):
        occupation=3 
    elif (option3=="Construction worker"):
        occupation=0 
    else:
        occupation=1

    option17 = st.selectbox("workplace hazards: ",("",'None','Chemicals','Other','Radiation','Heavy machinery','Noise','Biological hazards'),key="workplace_hazard")
    if (option17=='None'):
        workplace_hazards=4 
    elif (option17=="Chemicals"):
        workplace_hazards=1 
    elif (option17=="Other"):
        workplace_hazards=5 
    elif (option17=="Radiation"):
        workplace_hazards=6 
    elif (option17=="Heavy machinery"):
        workplace_hazards=2 
    elif (option17=="Noise"):
        workplace_hazards=3 
    else:
        workplace_hazards=0
    

    option6 = st.selectbox("karyotyping method",("",'Standard','Array CGH','M-FISH','FISH','Spectral'),key="karyotyping")
    if (option6=='Standard'):
        karyotyping_method=4
    elif (option6=="Array CGH"):
        karyotyping_method=0
    elif (option6=="M-FISH"):
        karyotyping_method=2
    elif (option6=="FISH"):
        karyotyping_method=1
    else:
        karyotyping_method=3




    option7 = st.selectbox("sample type",("",'Tumor tissue','Chorionic villus sample','Fetal blood','Saliva','Skin biopsy','Bone marrow','Placental tissue','Amniotic fluid','Sperm','Peripheral blood'),key="sample")
    if (option7=='Tumor tissue'):
        sample_type=9
    elif (option7=="Chorionic villus sample"):
        sample_type=2
    elif (option7=="Fetal blood"):
        sample_type=3
    elif (option7=="Saliva"):
        sample_type=6
    elif (option7=="Skin biopsy"):
        sample_type=7
    elif (option7=="Bone marrow"):
        sample_type=1
    elif (option7=="Placental tissue"):
        sample_type=5
    elif (option7=="Amniotic fluid"):
        sample_type=0
    elif (option7=="Sperm"):
        sample_type=8
    else:
        sample_type=4



    option8 = st.selectbox("chromosome morphology",("",'Acrocentric chromosome','Telocentric chromosome','Metacentric chromosome','Submetacentric chromosome'),key="chromosome_morph")
    if (option8=='Acrocentric chromosome'):
        chromosome_morphology=0
    elif (option8=="Telocentric chromosome"):
        chromosome_morphology=3
    elif (option8=="Metacentric chromosome"):
        chromosome_morphology=1
    else:
        chromosome_morphology=2
    


    option9 = st.selectbox("chromosome_banding: ",("",'C-banding','T-banding','R-banding','Q-banding','G-banding'),key="chromosome_band")
    if (option9=='C-banding'):
        chromosome_banding=0
    elif (option9=="T-banding"):
        chromosome_banding=4
    elif (option9=="R-banding"):
        chromosome_banding=3
    elif (option9=="Q-banding"):
        chromosome_banding=2
    else:
        chromosome_banding=1

    

    
    
    chromosome_count = st.slider('How many chromosome count are present ?', 0, 59, key="chromosomecount")
    st.write(chromosome_count," chromosome count" )

    st.write("\n")
    st.write("\n")

    chromosome_location = st.slider('where is the chromosome located ?', 0, 24, key="chromosome_locationslide")
    if chromosome_location==24:
        chromosome_location=="Y"
        st.write("chromosome is located in", chromosome_location)

    if chromosome_location==0:
        chromosome_location="X"
        st.write("chromosome is located in ", chromosome_location)
    
    else:
        st.write("chromosome is located in ", chromosome_location)


    st.write("\n")
    st.write("\n")



    


    anomalyResult = ''#for displaying result
    
    # creating a button for Prediction
    if age!="" and option1!=""  and option2!=""  and option3!=""  and option4!="" and option5!="" and option6!="" and option7 !=""and  chromosome_count is not None and option8 !="" and option9!="" and chromosome_location is not None  and option13 !="" and option14 !="" and option15 !="" and option16 !="" and option17 !="" and st.button('Predict'):
        anomalyResult = chromosomeAnom([age,Gender,Ethnicity,occupation,medical_history,family_medical_history,karyotyping_method,sample_type,chromosome_count,chromosome_morphology,chromosome_location,chromosome_banding,telomere_length,medications_taken,lifestyle_habits,environmental_exposure,workplace_hazards])
        st.success(anomalyResult)


def multi(input_data):
    loaded_model=pk.load(open("chromosomeAnomaliesMultiPredict.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    
    st.header('A view of the uploaded dataset')
    st.markdown('')
    st.dataframe(dfinput)

    forLoanId=dfinput["patient_id"]
    
    dfinput=dfinput.drop(columns=["patient_id"],axis=1)
    X_getam=dfinput
    #replace some column values
    dfinput.replace({"chromosome_location": {'x':0,'y': 24}},inplace=True)


    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    columns=["gender","ethnicity","medical_history","family_history","karyotyping_method","sample_type","chromosome_morphology","chromosome_banding","telomere_length","medications_taken","lifestyle_habits","environmental_exposure","occupation","workplace_hazards"]
    for col in columns:
        dfinput[col]= label_encoder.fit_transform(dfinput[col])

  
    X_getam = np.asarray(X_getam)
    
    
    predict=st.button("predict")


    if predict:
        prediction = loaded_model.predict(X_getam)
        interchange=[]
        for i in prediction:
            if i==1:
                newi="Anomaly is present"
                interchange.append(newi)
            elif i==0:
                newi="No Anomaly is present"
                interchange.append(newi)
            
        st.subheader('Here is your prediction')
        prediction_output = pd.Series(interchange, name='Chromosome anomaly results')
        prediction_id = pd.Series(forLoanId)
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

if selection =="Predict for a Single-Patient":
    main()

if selection == "Predict for Multi-Patient":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your csv file here')
    uploaded_file = st.file_uploader("", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file).
        multi(uploaded_file)
    else:
        st.info('Upload your dataset !!')
    