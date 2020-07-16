# %% CNN Model Tester
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.image import pil_to_array
import PIL

import tempfile
from shutil import copyfile, rmtree

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


# Set working directory to the path of the current script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# %% Load in the pre-trained CNN model
model = tf.keras.models.load_model("./models/classifier_v1")

global randNum
randNum = np.random.randint(1000)
temp_dir_path = "./data/tempDir" + str(randNum)

# %% Load in the uploaded image into a temporary directory
def loadImageIntoTempDir(img, img_type):
    temp_dir_path = "./data/tempDir" + str(randNum)

    if not os.path.isdir(temp_dir_path):
        os.makedirs(temp_dir_path + "/unlabelled/")

    image_filename = str(temp_dir_path) + "/unlabelled/img." + img_type

    with open(image_filename, "wb") as image_result:
        image_result.write(img)

    # Resize the image using pillow
    image = PIL.Image.open(image_filename)
    w, h = image.size
    image = image.resize(size=(int(w / h * 500), 500))
    image.save(image_filename)

    return temp_dir_path


# %% Create a generator object from the image in the temp dir
def createGenObject(temp_dir_path):
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    # Generator for our image
    temp_image_generator = ImageDataGenerator(rescale=1.0 / 255,)

    temp_data_gen = temp_image_generator.flow_from_directory(
        batch_size=1,
        directory=temp_dir_path,
        shuffle=False,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="binary",
    )

    temp_image, _ = next(temp_data_gen)

    return temp_data_gen, temp_image


# %% Assign the prediction into a grade bin
def assignPrediction(temp_data_gen, model):
    preds_array = np.array([i[0] for i in model.predict(temp_data_gen)])
    grade = np.digitize(preds_array, bins=[-10, -2, 0.5, 1.5, 3, 4, 5, 10]) - 1

    return grade


# %% Visualise grade predictions
def assignImgGrade(img, grade):
    grades_ref = ["A+", "A", "B", "C", "D", "E", "F"]

    # Initiate an in-memory file of bytes
    buf = io.BytesIO()

    # Get the width and height of the image
    h = img.shape[0]
    w = img.shape[1]
    xypos = [w * 0.5, h * 0.9]

    # Create the final plot
    fig, ax = plt.subplots()

    ax.annotate(
        s="GRADE: " + str(grades_ref[grade[0]]),
        xy=xypos,
        bbox={
            "facecolor": "#303030",
            "alpha": 0.9,
            "pad": 0.3,
            "boxstyle": "round, pad=0.3",
            "lw": 0,
        },
        color="#fffae0",
        fontsize=24,
        fontname="Arial",
        fontweight="bold",
        ha="center",
    )

    ax.imshow(img)
    ax.axis("off")

    plt.tight_layout()

    plt.savefig(buf, format="png")  # save to the above file object

    data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
    plt.close()

    return "data:image/png;base64,{}".format(data)

    # return fig


# %% Define the app object
# external_stylesheets = ["./assets/style.css"]
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=external_stylesheets,
)
app.title = "Latte Art Classifier"

# %% Create the app layout
app.layout = html.Div(
    style={
        "verticalAlign": "middle",
        "padding": "20px 30px 20px 30px",
        "backgroundColor": "#fffcf2",
        "textAlign": "left",
        "margin": "0px",
        "top": "0px",
        "left": "0px",
        "bottom": "0px",
    },
    className="body",
    children=[
        # Title and logo
        dbc.Row(html.H1("Latte Art Classifier")),
        # Logo
        # html.Img(
        #     src="./assets/logo.png",
        #     style={
        #         "height": "5%",
        #         "width": "5%",
        #         "float": "left",
        #         "position": "relative",
        #         "margin": "10px",
        #     },
        # ),
        # Subtitle
        dbc.Row(children=[html.H3("A web app by HG")], className="row",),
        # Caption
        dbc.Row(html.P("☕ 💻", style={"font-size": "1.25em"})),
        dbc.Row(html.P("Upload your best attempt at a latte art!")),
        # html.Br(),
        # Upload image
        dbc.Row(
            dbc.Col(
                # html.Div(
                [
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Take a picture or ", html.A("Select File")]
                        ),
                        style={
                            "width": "90%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "margin": "0px",
                        },
                        # Allow multiple files to be uploaded
                        # multiple=True,
                    )
                ],
                # )
            )
        ),
        # html.Div(id="output-contents-b64"),
        # Output image and message
        dbc.Row(
            # html.Div(
            [
                html.Img(
                    id="output-data-upload",
                    style={"height": "40vh", "max-width": "95%"},
                )
            ],
            style={"height": "25%"}
            # )
        ),
        html.Br(),
        dbc.Row(html.P(id="msg1")),
    ],
)

# %% Write app callbacks

# Grading output
@app.callback(
    Output("output-data-upload", "src"),
    [Input("upload-data", "contents")],
    [State("upload-data", "filename"), State("upload-data", "last_modified")],
)
def getPredictionOfImage(content, names, dates, model=model):
    if content is not None:
        # content = "abcdeghifklmnop"
        img_type = str(content).split("/")[1].replace(";base64,", "")
        img_encoded = str(content).replace(
            str("data:image/" + str(img_type) + ";base64,"), ""
        )
        img_decoded = base64.b64decode(img_encoded)

        # Save the image into a temporary directory
        temp_dir_path = loadImageIntoTempDir(img=img_decoded, img_type=img_type)

        # Create a generator object based on the contents of the temporary directory
        temp_data_gen, temp_image = createGenObject(temp_dir_path)

        # Assign a grade to the image in the temp dir using the model
        grade = assignPrediction(temp_data_gen, model)

        # Generate the final plot with the predictedgrade
        fname = str(temp_dir_path + "/unlabelled/img." + img_type)
        # img_high_res = mpimg.imread(fname=fname, format=img_type)
        img_high_res = pil_to_array(PIL.Image.open(fname))

        finalPlot = assignImgGrade(img=img_high_res, grade=grade)

        # Clean up and remove the temporary directory
        rmtree(temp_dir_path)

        return finalPlot

    return None


# Msg1 output and changes
content_prev = None


@app.callback(
    Output("msg1", "children"),
    [Input("upload-data", "contents"), Input("output-data-upload", "src")],
)
def printMessage(content, img):
    global content_prev
    if content is not None and img is None and content is not content_prev:
        content_prev = content
        return "I am grading your latte art ..."

    if img is not None and content is not None:
        return "Art graded! Try grading another one."


# Write a callback to visualise the encoded b64 image
# @app.callback(
#     Output("output-contents-b64", "children"),
#     [Input("upload-data", "contents")],
#     [State("upload-data", "filename"), State("upload-data", "last_modified")],
# )
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = html.Pre(list_of_contents)
#         return children


# %% Run the server
server = app.server

if __name__ == "__main__":
    app.run_server(debug=False)
