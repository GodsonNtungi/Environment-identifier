import streamlit as st
from st_clickable_images import clickable_images
from pathlib import Path
import base64
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = keras.models.load_model('intelli1.h5')
st.header('Environment Identifier')
pictures = Path('seg_pred/seg_pred')
col1, col2= st.columns(2,gap='small')

picturepath = [[], []]
picturepath2 = [[], []]
picturepath3 = [[], []]
count = 0
for apath in pictures.glob('*.jpg'):
    count += 1
    if count < 11:
        with open(apath, 'rb') as image:
            encoded = base64.b64encode(image.read()).decode()
            picturepath[1].append(apath)
            picturepath[0].append(f'data:image/jpeg;base64,{encoded}')
    else:
        break
    # elif 4 < count < 8:
    #     with open(apath, 'rb') as image:
    #         encoded = base64.b64encode(image.read()).decode()
    #         picturepath2[1].append(apath)
    #         picturepath2[0].append(f'data:image/jpeg;base64,{encoded}')
    # elif 8 < count < 13:
    #     with open(apath, 'rb') as image:
    #         encoded = base64.b64encode(image.read()).decode()
    #         picturepath3[1].append(apath)
    #         picturepath3[0].append(f'data:image/jpeg;base64,{encoded}')

with col1:
    clicked1 = clickable_images(picturepath[0],
                                titles=[i for i in range(len(picturepath[0]))],
                                div_style={"display": "flex", "justify-content": "normal", "flex-wrap": "wrap"},
                                img_style={"margin": "5px", "height": "200px"},
                                )

# with col2:
#     clicked2 = clickable_images(picturepath2[0],
#                                 titles=[i for i in range(len(picturepath2[0]))],
#                                 div_style={"display": "flex", "justify-content": "start", "flex-wrap": "wrap"},
#                                 img_style={"margin": "5px", "height": "200px"},
#                                 )
#
# with col3:
#     clicked3 = clickable_images(picturepath3[0],
#                                 titles=[i for i in range(len(picturepath3[0]))],
#                                 div_style={"display": "flex", "justify-content": "start", "flex-wrap": "wrap"},
#                                 img_style={"margin": "5px", "height": "200px"},
#                                 )
result = []
environment = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: "mountain", 4: "sea", 5: "street", 6: "unknown"}
error_threshold = 0.1
# order the prediction output
with col2:
    st.markdown(f'##### Identify the type of environment ')
    if clicked1 > -1:  # or clicked2 > -1 or clicked3 > -1:
        with st.spinner('Wait for it...'):
            image = Image.open(picturepath[1][clicked1])
            image = image.resize((128, 128))
            image = np.array(image)
            prediction = model.predict([image[None, :, :, :]])
            print(str(prediction))
            results = [[i, r] for i, r in enumerate(prediction[0]) if r > error_threshold]
            clicked1 = -1
            results.sort(key=lambda x: x[1], reverse=True)
            time.sleep(2)
            st.success('Done')

            for result in results:
                st.markdown(f'**{environment[result[0]]}**')
                st.write(f'{round(result[1] * 100, 2)}%')
                value = float(round(result[1] , 2))
                st.progress(value)




        # if clicked2 > -1:
        #     image = Image.open(picturepath2[1][clicked2])
        #     image = image.resize((128, 128))
        #     image = np.array(image)
        #     prediction = model.predict([image[None, :, :, :]])
        #     print(str(prediction))
        #     results = [[i, r] for i, r in enumerate(prediction[0]) if r > error_threshold]
        #     clicked2 = -1

        # if clicked3 > -1:
        #     image = Image.open(picturepath3[1][clicked2])
        #     image = image.resize((128, 128))
        #     image = np.array(image)
        #     prediction = model.predict([image[None, :, :, :]])
        #     print(str(prediction))
        #     results = [[i, r] for i, r in enumerate(prediction[0]) if r > error_threshold]
        #     clicked3 = -1

    # else:
    #     results= [[6,0.1]]


