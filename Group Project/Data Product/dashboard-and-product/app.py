import  streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='TeleInsight - CCP App',
    page_icon="☎️",
    layout='wide',
    
)

if "toggle" not in st.session_state:
    st.session_state.toggle = False  # Default to on

if "toggle_key" not in st.session_state:
    st.session_state.toggle_key = 1
    
with open('./best_model_no_voicemail.pkl','rb') as f:
    best_model_no_voicemail = pickle.load(f)
    
with open('./best_model_voicemail_true.pkl','rb') as f:
    best_model_yes_voicemail = pickle.load(f) 
            
#Backend Model Prediction
def print_prediction(d,toggle_value): 
    
    if toggle_value:
        prediction=best_model_yes_voicemail.predict(d)
    else:
        prediction=best_model_no_voicemail.predict(d)
    if prediction==0:
        st.header(f"The Customer is DISSATSFIED☹️ with the service and will leave")
    elif prediction==1:
        st.header(f"The Customer is SATSFIED😊 with the service and will remain")
    else:
        st.title("Awaiting input")
        
def toggle_toggle():
    st.session_state.toggle = not st.session_state[st.session_state.toggle_key]
    st.session_state.toggle_key += 1

form_prediction=100
#Frontend Form

wide_page, narrow_page = st.columns([0.7,0.3])

with wide_page:
    st.title("Customer Churn Predictor App")
    internationalplan = st.selectbox("Does the customer have an international plan?",["Yes","No"])
    accountlength = st.number_input("Number of years customer has been in service?",1,250)
    numbercustomerservicecalls = st.number_input("Number of customer service calls made",0,9)
   
    col1, col2 = st.columns(2)
        
        #column 1
    with col1:
        totaldayminutes = st.number_input("Total day minutes",1,250)
        totaleveminutes = st.number_input("Total eve minutes",1,250)
        totalnightminutes = st.number_input("Total  Night minutes",1,250)
        totalintlminutes = st.number_input("Total Int Minutes",1,250)
        
        #column 2
    with col2:
        totaldaycalls = st.number_input("Total Day Calls",1,250)
        totalevecalls = st.number_input("Total Eve Calls",1,250)
        totalnightcalls = st.number_input("Total Night Calls",1,250)
        totalintlcalls = st.number_input("Total Intl Calls",1,250)
        

    toggle = st.toggle(
        "Toggle", value=st.session_state.toggle, key=st.session_state["toggle_key"]
        )
    
    if toggle:
        d=10
    else:
        d=0
        
    numbervmailmessages = st.number_input("Numer of voicemail messages",0,d)
            
            
    #Predict Function      
    internationalplan_converter = lambda x:1 if x=='Yes' else 0
    internationalplan = internationalplan_converter(internationalplan)
    if toggle:
        df=pd.DataFrame(
            {
                "accountlength":[accountlength],
                "internationalplan":[internationalplan],
                "numbervmailmessages": [numbervmailmessages],
                "totaldayminutes":[totaldayminutes],
                "totaldaycalls":[totaldaycalls],
                "totaleveminutes":[totaleveminutes],
                "totalevecalls":[totalevecalls],
                "totalnightminutes":[totalnightminutes],
                "totalnightcalls":[totalnightcalls],
                "totalintlminutes":[totalintlminutes],
                "totalintlcalls":[totalintlcalls],
                "numbercustomerservicecalls":[numbercustomerservicecalls]
            }
        )
    else:
                df=pd.DataFrame(
            {
                "accountlength":[accountlength],
                "internationalplan":[internationalplan],
                "totaldayminutes":[totaldayminutes],
                "totaldaycalls":[totaldaycalls],
                "totaleveminutes":[totaleveminutes],
                "totalevecalls":[totalevecalls],
                "totalnightminutes":[totalnightminutes],
                "totalnightcalls":[totalnightcalls],
                "totalintlminutes":[totalintlminutes],
                "totalintlcalls":[totalintlcalls],
                "numbercustomerservicecalls":[numbercustomerservicecalls]
            }
        )
                
    predict = st.button("Predict")
           
with narrow_page:
    st.title("Prediction")
    if  predict:
        print_prediction(d=df,toggle_value=toggle)
    else:
        st.header("Awaiting Orders")