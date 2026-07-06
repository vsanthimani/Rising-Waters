from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    Date
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import date

Base = declarative_base()

# -----------------------------
# User Table
# -----------------------------
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    user_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    password = Column(String(100))
    role = Column(String(50))

    # One User -> Many Weather Records
    weather_records = relationship("WeatherData", back_populates="user")

    def __repr__(self):
        return f"<User({self.user_name})>"


# -----------------------------
# Weather Data Table
# -----------------------------
class WeatherData(Base):
    __tablename__ = "weather_data"

    data_id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.user_id"))

    annual_rainfall = Column(Float)
    cloud_visibility = Column(Float)
    seasonal_rainfall = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)

    # Many Weather Records -> One User
    user = relationship("User", back_populates="weather_records")

    # One Weather Record -> One Prediction
    prediction = relationship(
        "Prediction",
        uselist=False,
        back_populates="weather_data"
    )

    def __repr__(self):
        return f"<WeatherData(ID={self.data_id})>"


# -----------------------------
# Machine Learning Model Table
# -----------------------------
class MachineLearningModel(Base):
    __tablename__ = "ml_models"

    model_id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    algorithm_type = Column(String(100))
    accuracy = Column(Float)
    model_file = Column(String(200))

    # One Model -> Many Predictions
    predictions = relationship("Prediction", back_populates="model")

    def __repr__(self):
        return f"<Model({self.model_name})>"


# -----------------------------
# Prediction Table
# -----------------------------
class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True)

    data_id = Column(Integer, ForeignKey("weather_data.data_id"), unique=True)
    model_id = Column(Integer, ForeignKey("ml_models.model_id"))

    prediction_result = Column(String(50))
    flood_probability = Column(Float)
    prediction_date = Column(Date)

    # Relationships
    weather_data = relationship("WeatherData", back_populates="prediction")
    model = relationship("MachineLearningModel", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction({self.prediction_result})>"


# ==========================================================
# Database Creation
# ==========================================================

engine = create_engine("sqlite:///flood_prediction.db")

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# ==========================================================
# Insert Sample Data
# ==========================================================

# User
user = User(
    user_name="Admin",
    email="admin@gmail.com",
    password="admin123",
    role="Administrator"
)

session.add(user)
session.commit()

# Weather Data
weather = WeatherData(
    user_id=user.user_id,
    annual_rainfall=1250.5,
    cloud_visibility=70,
    seasonal_rainfall=450.2,
    temperature=29.5,
    humidity=82
)

session.add(weather)
session.commit()

# ML Model
model = MachineLearningModel(
    model_name="Random Forest",
    algorithm_type="Random Forest Classifier",
    accuracy=96.45,
    model_file="random_forest.pkl"
)

session.add(model)
session.commit()

# Prediction
prediction = Prediction(
    data_id=weather.data_id,
    model_id=model.model_id,
    prediction_result="Flood Likely",
    flood_probability=0.92,
    prediction_date=date.today()
)

session.add(prediction)
session.commit()

print("Database Created Successfully!")

# ==========================================================
# Display Data
# ==========================================================

print("\n----- Users -----")
for u in session.query(User).all():
    print(u.user_id, u.user_name, u.email, u.role)

print("\n----- Weather Data -----")
for w in session.query(WeatherData).all():
    print(
        w.data_id,
        w.annual_rainfall,
        w.temperature,
        w.humidity
    )

print("\n----- ML Models -----")
for m in session.query(MachineLearningModel).all():
    print(
        m.model_name,
        m.algorithm_type,
        m.accuracy
    )

print("\n----- Predictions -----")
for p in session.query(Prediction).all():
    print(
        p.prediction_result,
        p.flood_probability,
        p.prediction_date
    )