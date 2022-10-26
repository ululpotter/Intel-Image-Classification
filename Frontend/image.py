import streamlit as st
import requests
import numpy as np
import json
from skimage.transform import resize
from PIL import Image


# Load page
def run():
    # widget input
    with st.form(key='form_parameters'):
        st.title("Intel Image Classification")
        uploaded_file = st.file_uploader("Choose a file", type='jpg')
        
        st.markdown('---')
        submitted = st.form_submit_button('Predict')
    
    if submitted:
        image = Image.open(uploaded_file)
        np_img = np.asarray(image)
        resized = resize(np_img, (150,150),anti_aliasing=True)
        # st.write(resized.shape)
        x = np.expand_dims(resized, axis=0)
        images = np.vstack([x])
        img_list = images.tolist()
        # model input
        input_data_json = json.dumps({
            'signature_name': 'serving_default',
            'instances': img_list
        })

        # inference
        URL = "https://image-class-ulul-azmi.herokuapp.com/v1/models/cv_model:predict"

        r = requests.post(URL, data=input_data_json)

        if r.status_code == 200:
            res = r.json()
            # st.write(res['predictions'][0][0])
            result_max_proba = res.argmax(axis=-1)[0]
            label_dict = {0:'buildings',1:'forest',2:'glacier',3:'mountain',4:'sea',5:'street'}
            result_class = label_dict[result_max_proba]

            print('Result     : ', res)
            print('Max Class  : ', result_max_proba)
            print('Class Name : ', result_class)
            print('')
        else:
            st.write('Error')
        
        st.image(resized)
    

if __name__ == '__main__':
    run()

