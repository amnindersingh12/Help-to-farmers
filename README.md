# Help to farmers

## Link to the Demo
https://helptofarmers.herokuapp.com/

# PRESENTATION
![Presentation](https://i.ibb.co/23JVN8W/Screenshot-from-2021-07-15-17-39-38.png)
[Link to the presentation ](https://www.canva.com/design/DAEeiw1W8hw/v_eZ2YLwaqe3fZYdePt3cA/view?utm_content=DAEeiw1W8hw&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink#1)


In this project we have presented the idea that how we can use ML/DL into farming. In this project, we presents a website in which the following applications are implemented
* Plant disease prediction
* Crop recommendation
* Fertilizer recommendation


## Local Setup
1. Clone the repository:
  ***git clone https://github.com/amnindersingh12/Help-to-farmers.git***
2. Install dependencies:
  ***pip install -r requirements.txt***
3. Set your weather API key (OpenWeather):
  ***export WEATHER_API_KEY=your_api_key_here***
4. Run the app:
  ***python app.py***

## Improvements Implemented
* Numeric form validation with friendly errors on crop and fertilizer forms.
* Disease upload validation with clear user messages for missing/invalid files.
* Lazy model loading and startup checks with explicit missing-file errors.
* Structured logging for production troubleshooting.
* Pinned dependency versions in ***requirements.txt***.
* Gunicorn production configuration in ***gunicorn.conf.py***.
* Automated endpoint tests in ***tests/test_endpoints.py***.

## Run Tests
***pytest -q***

## Production Hosting (VM)
* Service: ***help-to-farmers-web.service***
* App bind: ***127.0.0.1:5002***
* Public route: ***https://alreadytaken.me/projects/help-to-farmers/live/***
* Project page: ***https://alreadytaken.me/projects/help-to-farmers/***


