import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)

from werkzeug.security import generate_password_hash, check_password_hash

from diabetes_model import load_or_train_bundle, predict_percent

# ------------------- App Setup -------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "instance" / "app.db"
BASE_DIR.joinpath("instance").mkdir(exist_ok=True)

app = Flask(__name__, instance_relative_config=True)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH.as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ------------------- Database Models -------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    pregnancies = db.Column(db.Integer)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    dpf = db.Column(db.Float)  # DiabetesPedigreeFunction
    age = db.Column(db.Integer)
    percent = db.Column(db.Float)
    label = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# Load ML Model at Startup (training happens automatically if needed)
BUNDLE = load_or_train_bundle()


# ------------------- Routes -------------------
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")

        if not email or not password:
            flash("Email and Password are required", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email is already registered", "error")
            return redirect(url_for("register"))

        user = User(email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("dashboard"))

        flash("Invalid login details", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    last_pred = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).first()
    return render_template("dashboard.html", last_pred=last_pred)


@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    if request.method == "POST":
        data = {
            "Pregnancies": int(request.form.get("Pregnancies", 0)),
            "Glucose": float(request.form.get("Glucose", 0)),
            "BloodPressure": float(request.form.get("BloodPressure", 0)),
            "SkinThickness": float(request.form.get("SkinThickness", 0)),
            "Insulin": float(request.form.get("Insulin", 0)),
            "BMI": float(request.form.get("BMI", 0)),
            "DiabetesPedigreeFunction": float(request.form.get("DiabetesPedigreeFunction", 0)),
            "Age": int(request.form.get("Age", 0)),
        }

        percent = predict_percent(BUNDLE, data)
        label = 1 if percent >= 50 else 0  # 50% threshold

        record = Prediction(user_id=current_user.id, percent=percent, label=label, **{
            "pregnancies": data["Pregnancies"],
            "glucose": data["Glucose"],
            "blood_pressure": data["BloodPressure"],
            "skin_thickness": data["SkinThickness"],
            "insulin": data["Insulin"],
            "bmi": data["BMI"],
            "dpf": data["DiabetesPedigreeFunction"],
            "age": data["Age"]
        })
        db.session.add(record)
        db.session.commit()

        return render_template("predict.html", result=percent, label=label, form=data)

    return render_template("predict.html", result=None)


@app.route("/history")
@login_required
def history():
    rows = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    labels = [r.created_at.strftime("%d %b %Y %H:%M") for r in rows][::-1]
    values = [round(r.percent,2) for r in rows][::-1]
    return render_template("history.html", rows=rows, labels=labels, values=values)


if __name__ == "__main__":
    app.run(debug=True)
