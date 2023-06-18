import streamlit as st
import AutoRegression as AR
import pandas as pd
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("# Regression")
st.sidebar.markdown("# Regression")

if "regression_disabled" not in st.session_state:
    st.session_state.regression_disabled = True

model_list = pd.read_csv("./pages/model_list.csv",index_col=0)
only_id = list(model_list['ID'])
metric = pd.read_csv("./pages/metrics.csv",index_col=0)
metrics = list(metric['ID'])
r_data_=None

def uploader_callback():
    if st.session_state.regression_disabled == False:
        st.session_state.regression_disabled = True

r_data_ = st.file_uploader(
    label='regression file upload',
    on_change=uploader_callback,
    key='regression file_uploader'
)


method_box=[]
analysis_data = None
single_model_box=[]
check_model = False
setup_ready = False

    
if r_data_ is not None:
    r_data = pd.read_csv(r_data_)
    st.subheader('Original Data Head')
    with st.expander("Start"):
        st.write(r_data.head())
        
    st.subheader('Data Pre-processing')
    with st.expander("Start"):
        missing_cols,missing_series = AR.search_missing_value(r_data)
        if len(missing_cols) != 0 and len(missing_series) != 0:
            missing_check = 1
        else:
            missing_check = 0
        st.subheader('Missing Value')
        if missing_check == 1:
            st.write(missing_cols)
            st.write(missing_series)
            for i in missing_cols:
                method = st.selectbox(f"Variable: {i}",["linear", "index","pad","nearest"],key=i)
                method_box.append(method)
        else:
            st.write("Not Found Missing Value")
        
        analysis_data = AR.interpolation(r_data, missing_cols, method_box) 
        
        target_ = st.selectbox('Select Target', list(analysis_data.columns), len(analysis_data.columns)-1)
        gpu = st.radio(label = 'GPU 사용 여부', options = ['True', 'False'])
        out = st.radio(label = '이상치 제거 여부', options = ['True', 'False'])
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) 
        st.write(model_list)
        single_model_setup = st.selectbox('원하는 모델의 ID를 선택해주세요', only_id)
        train = st.slider('Train Data 비율', 0, 100, 70, 10)
        train = train/100
        target_model_arr = st.multiselect('비교할 모델들을 선정해주세요.', only_id)
        optimize_setup = st.selectbox('Optimize 기준을 정해주세요. Optimize는 Random Grid Search Algorithm을 사용합니다.', metrics)

        if st.checkbox('Show Data header'):
            st.subheader('Data header')
            if analysis_data is not None:
                st.write(analysis_data.head())
            else:
                st.write(r_data.head())

        btn_clicked = st.button("Confirm", key='confirm_btn')

        if btn_clicked:
            con = st.container()
            con.caption("Ready!")
            st.session_state.regression_disabled = False

if analysis_data is not None:
    st.subheader('Data Visiaulize')
    with st.expander("Start"):      
        te_data = analysis_data.copy()
        #te_data["Time"] = te_data["Time"].str.replace("오전", "AM").str.replace("오후", "PM")
        #te_data["Time"] = pd.to_datetime(te_data["Time"], format="%p %I:%M:%S").dt.strftime("%H:%M:%S")
        #te_data["DateTime"] = pd.to_datetime(te_data["Date"] + " " + te_data["Time"])
        #te_data = te_data.drop(['Time', 'Date'], axis=1)
        option = st.selectbox(
            'Select Variable', 
            (te_data.columns),
            disabled=st.session_state["regression_disabled"]
            )
        show = st.button("Show", disabled=st.session_state["regression_disabled"])
        if show:
            st.line_chart(te_data[option])

if analysis_data is not None:
    st.subheader('Setup')
    with st.expander("Start"):  
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) 

        confirm = st.button("confirm", disabled=st.session_state.regression_disabled, key="confirm_btn2")
        if confirm:
            check_model = True
        if check_model == True:
            progress_bar = st.progress(0)
            setting = AR.setup(analysis_data, target_, train, bool(gpu), bool(out))
            for percent_complete in range(100):
                time.sleep(0.1)  
                progress_bar.progress(percent_complete + 1)
            con = st.container()
            con.caption("Done")
            setting_result = AR.save_df()
            st.write(setting_result)
try:
    if setting_result is not None:
        st.subheader('%s Model Result'%single_model_setup)
        with st.expander("Start"):
            progress_bar2 = st.progress(0)
            single_model = AR.single(single_model_setup)
            for percent_complete2 in range(100):
                time.sleep(0.1)  
                progress_bar2.progress(percent_complete2 + 1)
            single_result = AR.save_df()
            st.markdown("### Result")
            st.write(single_result.drop(['Mean', 'Std'], axis=0).style.highlight_min(axis=0))
            single_visual_graph = AR.single_visual(single_result.drop(['Mean', 'Std'], axis=0))
            single_visual_graph = AR.save_df()
            st.markdown("### Graph")
            st.line_chart(single_visual_graph.drop(['Mean', 'Std'], axis=0))
        st.subheader('Compare Model Result')
        with st.expander("Start"):
            progress_bar3 = st.progress(0)
            best_model = AR.compare(target_model_arr)
            for percent_complete3 in range(100):
                time.sleep(0.1)  
                progress_bar3.progress(percent_complete3 + 1)
            compare_result = AR.save_df()
            st.markdown("### Compare All Models")
            st.write(compare_result.style.highlight_min(axis=0))
        st.subheader('Optimize Best Model')
        with st.expander("Start"):
            st.markdown("### Optimize Result")
            st.markdown("#### 만약 Opimize 결과가 좋지 않으면 기존 모델을 반환합니다.")
            with st.spinner("Optimize Your Model"):
                optimize_best_model = AR.tune(best_model, optimize_setup)
                opt_result = AR.save_df()
                st.write(opt_result.drop(['Mean', 'Std'], axis=0).style.highlight_min(axis=0))
        st.subheader('Predcit Result')
        with st.expander("Start"):
            pred1 = AR.prediction(single_model)
            pred1_result = AR.save_df()
            pred2 = AR.prediction(best_model)
            pred2_result = AR.save_df()
            pred3 = AR.prediction(optimize_best_model)
            pred3_result = AR.save_df()
            st.markdown("#### %s model Prediction Result"%single_model_setup)
            st.write(pred1_result)
            st.markdown("#### best model Prediction")
            st.write(pred2_result)
            st.markdown("#### optimize model Prediction")
            st.write(pred3_result)
        st.subheader('Save Model')
        with st.expander("Start"):
            AR.save_model(single_model,'./pages/models/%s'%single_model_setup)
            AR.save_model(best_model,'./pages/models/best_model')
            AR.save_model(optimize_best_model,'./pages/models/opt_model')
            con = st.container()
            con.caption("성공적으로 저장하였습니다.")
            con.caption(" Relative path : ./pages/models/")

except:
    st.write("Setup을 완료하면 자동으로 실행됩니다.")