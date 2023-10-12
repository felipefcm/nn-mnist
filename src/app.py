import gradio as gr
import pickle

with open('model_91_7248.bin', 'rb') as f:
    nn = pickle.load(f)


def predict(input):
    x = input.reshape((784, 1))
    p = nn.feed_forward(x).reshape((10,))

    return dict(enumerate(p))


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Sketchpad(
            shape=(28, 28),
            brush_radius=1.2,
        )
    ],
    outputs=[
        gr.Label(
            num_top_classes=3,
            scale=2,
        )
    ],
    live=True,
    allow_flagging=False
).launch()
