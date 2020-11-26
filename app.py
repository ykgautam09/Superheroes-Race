from flask import Flask, request, render_template
from model import get_hero_race

app = Flask(__name__)


@app.route('/')
def nlp_route():
    return render_template('superheroRace.html', size=0)


@app.route('/', methods=['POST'])
def cosine_model():
    history_text = request.form.get('history_text')
    hero_race = get_hero_race(str(history_text))
    return render_template('superheroRace.html', size=len(hero_race), race=hero_race)


if __name__ == '__main__':
    app.run(port=process.env.PORT )
