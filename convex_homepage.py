# conda activate thesis_3.11 && cd github/msc_thesis && streamlit run convex_homepage.py

# Custom imports
from streamlit_initialise import *

# Page configurations
st.set_page_config(layout = "wide")
screen_height = ScreenData().st_screen_data(key="screen_stats_")["innerHeight"]

button_images = []
for filename in ["syringe", "dengue", "obesity"]:
    with open(f"images/{filename}.png", "rb") as file:
        button_images += [read_img(file)]

# <img src="data:image/png;base64,{logo}" style="width: 200px; height: 200px;">
st.markdown(f'''
            <div style="display: flex; align-items: center; gap: 10px; max-width: fit-content; margin-inline: auto">
            <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0"/>
            <span class="material-symbols-rounded" style="font-size: 100px; color: #B2A8D3;">cognition</span>
            <h1 style="font-size: 4em; margin: 0; width: fit-content;">ConveX</h1>
            </div>
            ''', unsafe_allow_html = True)

# st.markdown('''
#             <div style="display: flex; align-items: center; max-width: fit-content; margin-inline: auto">
#             <center>
#             <h3 style="font-size: 1.5em; margin: 0; width: fit-content;">Your personalised Conversational eXplainer</h3>
#             </center>
#             </div>
#             ''', unsafe_allow_html = True)

# /* On hover, add a black background color with a little bit see-through */
# .prev:hover, .next:hover {
# background-color: rgba(0,0,0,0.8);
# }

# /* The dots/bullets/indicators */
# .dot {
# cursor: pointer;
# height: 15px;
# width: 15px;
# margin: 0 2px;
# background-color: #bbb;
# border-radius: 50%;
# display: inline-block;
# transition: background-color 0.6s ease;
# }

# .active, .dot:hover {
# background-color: #717171;
# } 

components.html(
    f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name = "viewport" content = "width=device-width, initial-scale = 1">
    <style>
    * {{box-sizing: border-box;}}
    body {{font-family: Verdana, sans-serif;}}
    .mySlides {{display: none;}}
    img {{vertical-align: middle;}}

    /* Slideshow container */
    .slideshow-container {{
    max-width: 1000px;
    position: relative;
    margin: auto;
    }}

    /* Next & previous buttons */
    .prev, .next {{
    cursor: pointer;
    position: absolute;
    top: 50%;
    width: auto;
    padding: 16px;
    margin-top: -22px;
    color: white;
    font-weight: bold;
    font-size: 18px;
    transition: 0.6s ease;
    border-radius: 0 3px 3px 0;
    user-select: none;
    }}

    /* Position the "next button" to the right */
    .next {{
    right: 0;
    border-radius: 3px 0 0 3px;
    }}

    /* Fading animation */
    .fade {{
    animation-name: fade;
    animation-duration: 1.5s;
    }}

    @keyframes fade {{
    from {{opacity: .3}}
    to {{opacity: 1}}
    }}

    /* On smaller screens, decrease text size */
    @media only screen and (max-width: 300px) {{
    .text {{font-size: 11px}}
    }}
    </style>
    </head>
    <body>

    <div class="slideshow-container">

    <div class="mySlides fade">
    <center>
    <h3 style="font-size: 1.5em; margin: 0; width: fit-content; color: white">Your personalised Conversational eXplainer</h3>
    </center>
    </div>

    <div class="mySlides fade">
    <center>
    <h3 style="font-size: 1.5em; margin: 0; width: fit-content; color: white">Providing the most up-to-date selection of popular LLMs</h3>
    </center>
    </div>

    <div class="mySlides fade">
    <center>
    <h3 style="font-size: 1.5em; margin: 0; width: fit-content; color: white">Read more about ConveX <a href = "https://github.com/HenryElJ/msc_thesis"; style = "color: white"; target = "_blank">here</a></h3>
    </center>
    </div>

    <a class="prev" onclick="plusSlides(-1)">&#10094;</a>
    <a class="next" onclick="plusSlides(1)">&#10095;</a>
    </div>
    <br>

    <div style="text-align:center">
      <span class="dot" onclick="currentSlide(1)"></span> 
      <span class="dot" onclick="currentSlide(2)"></span> 
      <span class="dot" onclick="currentSlide(3)"></span> 
    </div>

    <script>
    let slideIndex = 1;
    showSlides(slideIndex);

    // Next/previous controls
    function plusSlides(n) {{
      showSlides(slideIndex += n);
    }}

    // Thumbnail image controls
    function currentSlide(n) {{
      showSlides(slideIndex = n);
    }}

    function showSlides(n) {{
      let i;
      let slides = document.getElementsByClassName("mySlides");
      let dots = document.getElementsByClassName("dot");
      if (n > slides.length) {{slideIndex = 1}}
      if (n < 1) {{slideIndex = slides.length}}
      for (i = 0; i < slides.length; i++) {{
        slides[i].style.display = "none";
      }}
      for (i = 0; i < dots.length; i++) {{
        dots[i].className = dots[i].className.replace(" active", "");
      }}
      slides[slideIndex-1].style.display = "block";
      dots[slideIndex-1].className += " active";
    }}
    </script>

    </body>
    </html>
    """,
    height = 40,
)

# <img src="data:image/png;base64,{llm_images[0][1]}" style="width:3%">
# <img src="data:image/png;base64,{llm_images[3][1]}" style="width:3%">
# <img src="data:image/png;base64,{llm_images[6][1]}" style="width:3%">
# <img src="data:image/png;base64,{llm_images[1][1]}" style="width:3%">
# <img src="data:image/png;base64,{llm_images[4][1]}" style="width:3%">

_, divider, _ = st.columns([0.1, 0.8, 0.1]); divider.divider()
h1n1, dengue, obesity = st.columns(3)
container_height = screen_height - 380
img_width = container_height - 185

st.markdown('''<style>.convex-link {text-decoration: none !important; color: inherit !important}.convex-link:hover {color: #B2A8D3 !important}</style>''',  unsafe_allow_html = True)

with h1n1:
    st.markdown('''<center><a href = 'http://localhost:8501/h1n1' target = '_self' class = 'convex-link'><h2>H1N1 Virus</h2></a></center>''', unsafe_allow_html = True)
    h1n1_container = st.container(height = container_height, border = True)
    h1n1_container.write('''
                         Binary classification task predicting vaccine uptake for the H1N1 flu, using protected, behavioural and additional characteristics collected
                         from the 2009 H1N1 Flu Survey.
                         ''')
    if container_height < 350:
      with h1n1_container:
        col1_1, col1_2 = st.columns(2, vertical_alignment = "center")
      # https://unsplash.com/photos/a-hand-in-a-blue-glove-holding-a-syquet-1M0JM0z3GFA
        col1_1.markdown(f"<center><a href = 'http://localhost:8501/h1n1' target = '_self'><img src='data:image/png;base64,{button_images[0]}' width='{img_width + 70}'></a></center>", unsafe_allow_html = True)
        if col1_2.button("Click here to explore", key = "h1n1_button_1", use_container_width = True):
            st.switch_page("pages/h1n1.py")
    else:
      h1n1_container.markdown(f"<center><a href = 'http://localhost:8501/h1n1' target = '_self'><img src='data:image/png;base64,{button_images[0]}' width='{img_width}'></a></center>", unsafe_allow_html = True)
      h1n1_container.write("")
      if h1n1_container.button("Click here to explore", key = "h1n1_button_2", use_container_width = True):
          st.switch_page("pages/h1n1.py")

with dengue:
    st.markdown('''<center><a href = 'http://localhost:8501/dengue' target = '_self' class = 'convex-link'><h2>Dengue Fever</h2></a></center>''', unsafe_allow_html = True)
    dengue_container = st.container(height = container_height, border = True)
    dengue_container.write('''
             Regression task predicting the number of weekly dengue fever cases in San Juan (Puerto Rico), and Iquitos (Peru), 
             using environmental data collected from US Federal Gov. Agencies.
             ''')

    if container_height < 350:
      with dengue_container:
        # https://pixabay.com/photos/mosquito-insect-mosquito-bite-49141/
        col2_1, col2_2 = st.columns(2, vertical_alignment = "center")
        col2_1.markdown(f"<center><a href = 'http://localhost:8501/dengue' target = '_self'><img src='data:image/png;base64,{button_images[1]}' width='{img_width + 70}'></a></center>", unsafe_allow_html = True) 
        if col2_2.button("Click here to explore", key = "dengue_button_1", use_container_width = True):
            st.switch_page("pages/dengue.py")
    else:
      dengue_container.markdown(f"<center><a href = 'http://localhost:8501/dengue' target = '_self'><img src='data:image/png;base64,{button_images[1]}' width='{img_width}'></a></center>", unsafe_allow_html = True) 
      dengue_container.write("")
      if dengue_container.button("Click here to explore", key = "dengue_button_2", use_container_width = True):
          st.switch_page("pages/dengue.py")

with obesity:
    st.markdown("<center><a href = 'http://localhost:8501/obesity' target = '_self' class = 'convex-link'><h2>Obesity</h2></a></center>", unsafe_allow_html = True)
    obesity_container = st.container(height = container_height, border = True)
    obesity_container.write('''
             Multi-class classification task (clustering) of obesity levels in individuals from the countries Mexico, Peru and Colombia, using survey data of eating habits and physical condition.
             ''')    
    
    if container_height < 350:
      with obesity_container:
        # https://www.freepik.com/free-photo/view-tape-measure-with-apple-fruit_40460642.htm
        col3_1, col3_2 = st.columns(2, vertical_alignment = "center")
        col3_1.markdown(f"<center><a href = 'http://localhost:8501/obesity' target = '_self'><img src='data:image/png;base64,{button_images[2]}' width='{img_width + 70}'></a></center>", unsafe_allow_html = True) 
        if col3_2.button("Click here to explore", key = "obesity_button_1", use_container_width = True):
            st.switch_page("pages/obesity.py")
    else:
      obesity_container.markdown(f"<center><a href = 'http://localhost:8501/obesity' target = '_self'><img src='data:image/png;base64,{button_images[2]}' width='{img_width}'></a></center>", unsafe_allow_html = True) 
      obesity_container.write("")
      if obesity_container.button("Click here to explore", key = "obesity_button_2", use_container_width = True):
         st.switch_page("pages/obesity.py")

# Page configurations
st.markdown(f"<style>.block-container {{padding-top: 0.5rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; overflow: hidden}}</style>", unsafe_allow_html = True)