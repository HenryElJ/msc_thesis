def generate_tab(tab):
    return tab.replace(";\n", ";")


def add_chatbox_col(tab):
    return f'''with {tab}_ai:
        \tchat_container = st.container(height = screen_height - chat_padding, key = "{tab}_container");
    
        \twith chat_container:
            \t\tfor message in st.session_state.messages:
                \t\t\twith st.chat_message(name = message["name"], avatar =  message["avatar"]): 
                    \t\t\t\tst.write(message["content"]);
    
        \tif query := st.chat_input("Ask me a question!", accept_file = True, key = "{tab}_chat_input"):
            \t\tif query["text"] is None:
                \t\t\tpass
            \t\telse:
                \t\t\twith chat_container:
                    \t\t\t\twith st.chat_message(name = "user", avatar = user_avatar):
                        \t\t\t\t\tst.write(query["text"]); st.session_state.messages.append({{"name": "user", "avatar": user_avatar, "content": query["text"]}})
                    \t\t\t\twith st.chat_message(name = "assistant", avatar = ai_avatar):
                        \t\t\t\t\t\tmessage_input = [{{"type": "text", "text": query["text"]}}]; responses = lorem.paragraphs(10); st.write(responses); st.session_state.responses = responses; st.session_state.messages.append({{"name": model_selection, "avatar": ai_avatar, "content": st.session_state.responses}});
        
        \tst.info("Large language models can make mistakes. Please verify information before decisions.", icon = ":material/info:")'''

introduction_tab = '''intro1, intro2 = st.columns([0.25, 0.75], vertical_alignment = "center");
intro1.image("images/vaccine.jpeg");
intro2.markdown(
    """
    ### **Overview**

    [Immunisation](https://www.who.int/health-topics/vaccines-and-immunization#tab=tab_1) :link: is a fundamental aspect of primary health care and an indisputable human right. 
    Vaccines are critical to the prevention and control of infectious disease outbreaks, and underpin global health security.

    Vaccines work by training your immune system to create antibodies and assist your body’s natural defences to help build protection.
    Because vaccines contain only killed or weakened forms of germs like viruses or bacteria, they do not cause the disease or put you at risk of its complications.

    There are vaccines for more than [20 life-threatening diseases](https://www.who.int/teams/immunization-vaccines-and-biologicals/diseases) :link: including: cholera, typhoid, influenza, rabies, measles, mumps and rubella (MMR). 
    Immunisation against these diseases [prevents 3-3.5 million deaths each year](https://www.who.int/health-topics/vaccines-and-immunization#tab=tab_1) :link:.

    However, vaccines have always been a polarising topic. [Anti-vaccinationism](https://en.wikipedia.org/wiki/Vaccine_hesitancy) :link: (commonly known today as "anti-vax" or "anti-vaxxers"), refers to the complete opposition of vaccines - 
    often propogated through conspiracy theories, mis/dis information and fringe science. Such vaccine hesitancy has led to increasingly large numbers of the population to 
    delay, or outright refuse vaccinations, which as a result prevents heard immunity and leads to increased outbreaks and death from these diseases. 

    Vaccine hesitancy is characterised by the World Health Organisation as one of the [top-10 global health threats](https://www.who.int/news-room/spotlight/ten-threats-to-global-health-in-2019) :link:.
    """);
st.markdown("""---""");
virus1, virus2 = st.columns([0.75, 0.25], vertical_alignment = "top");
virus1.markdown(
    f"""
    ### **Task Outline**
    
    In the beginning of spring 2009, a major respiratory disease pandemic caused by the [H1N1 influenza virus](https://en.wikipedia.org/wiki/Influenza_A_virus_subtype_H1N1) :link:, colloquially named "swine flu," swept across the world. 
    Researchers estimate that in the first year, it was responsible for between [151,000 to 575,000 deaths globally](https://zenodo.org/records/1260250) :link:. 
    A vaccine for the H1N1 flu virus quickly followed and became publicly available in October 2009. 
    
    In late 2009 and early 2010, the United States conducted the [National 2009 H1N1 Flu Survey](https://www.cdc.gov/nchs/nis/data_files_h1n1.htm) :link:. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. 
    These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. 

    Using these features, a model has been developed to predict how likely an individual is to get vaccinated against the H1N1 influenza virus and seasonal flu.
    
    ---

    ### **Objective**

    This explainer dashboard aims to provide guidance for future public health efforts and increase the understanding of how these 
    protected, behavioural and opinion-based characteristics are associated with personal vaccination patterns.

    Using this dashboard, you will be able to determine:

    ##### **Modelling**
    - How accurately the model can predict if an individual receives their H1N1 and seasonal flu vaccines
        
    - Which features are most important in the model's predictions, and how do the predictions change as these feature values change

    - How fair and unbiased the model is
    
    ##### **Data**

    - Is the composition of the data suitable for modelling purposes

    <br>
    """, unsafe_allow_html = True);
virus2.image("images/h1n1.jpeg")'''

data_tab = '''data_dict, _ = st.columns(2);
data_dict.expander(":books: Data Dictionary").container(height = screen_height - dict_padding, border = False).markdown("""
##### **Data Labels**

Each row in the dataset represents one person who responded to the National 2009 H1N1 Flu Survey.

There are two target variables:

* **:red-background[h1n1_vaccine]** - Whether respondent received H1N1 flu vaccine.

* **:red-background[seasonal_vaccine]** - Whether respondent received seasonal flu vaccine.

Both are binary variables: 0 = No; 1 = Yes. Some respondents didn't get either vaccine, others got only one, and some got both. This is formulated as a multilabel (and not multiclass) problem.

##### **Features**

Provided is a dataset with 36 columns. The first column respondent_id is a unique and random identifier. The remaining 35 features are described below.

For all binary variables: 0 = No; 1 = Yes, 999 = Missing.

* **:red-background[h1n1_concern]** - Level of concern about the H1N1 flu.
0 = Not at all concerned; 1 = Not very concerned; 2 = Somewhat concerned; 3 = Very concerned.

* **:red-background[h1n1_knowledge]** - Level of knowledge about H1N1 flu.
0 = No knowledge; 1 = A little knowledge; 2 = A lot of knowledge.

* behavioral_antiviral_meds]** - Has taken antiviral medications. (binary)

* **:red-background[behavioral_avoidance]** - Has avoided close contact with others with flu-like symptoms. (binary)

* **:red-background[behavioral_face_mask]** - Has bought a face mask. (binary)

* **:red-background[behavioral_wash_hands]** - Has frequently washed hands or used hand sanitizer. (binary)

* **:red-background[behavioral_large_gatherings]** - Has reduced time at large gatherings. (binary)

* **:red-background[behavioral_outside_home]** - Has reduced contact with people outside of own household. (binary)

* **:red-background[behavioral_touch_face]** - Has avoided touching eyes, nose, or mouth. (binary)

* **:red-background[doctor_recc_h1n1]** - H1N1 flu vaccine was recommended by doctor. (binary)

* **:red-background[doctor_recc_seasonal]** - Seasonal flu vaccine was recommended by doctor. (binary)

* **:red-background[chronic_med_condition]** - Has any of the following chronic medical conditions: asthma or an other lung condition, diabetes, a heart condition, a kidney condition, sickle cell anemia or other anemia, a neurological or neuromuscular condition, a liver condition, or a weakened immune system caused by a chronic illness or by medicines taken for a chronic illness. (binary)

* **:red-background[child_under_6_months]** - Has regular close contact with a child under the age of six months. (binary)

* **:red-background[health_worker]** - Is a healthcare worker. (binary)

* **:red-background[health_insurance]** - Has health insurance. (binary)

* **:red-background[opinion_h1n1_vacc_effective]** - Respondent's opinion about H1N1 vaccine effectiveness.
1 = Not at all effective; 2 = Not very effective; 3 = Don't know; 4 = Somewhat effective; 5 = Very effective.

* **:red-background[opinion_h1n1_risk]** - Respondent's opinion about risk of getting sick with H1N1 flu without vaccine.
1 = Very Low; 2 = Somewhat low; 3 = Don't know; 4 = Somewhat high; 5 = Very high.

* **:red-background[opinion_h1n1_sick_from_vacc]** - Respondent's worry of getting sick from taking H1N1 vaccine.
1 = Not at all worried; 2 = Not very worried; 3 = Don't know; 4 = Somewhat worried; 5 = Very worried.

* **:red-background[opinion_seas_vacc_effective]** - Respondent's opinion about seasonal flu vaccine effectiveness.
1 = Not at all effective; 2 = Not very effective; 3 = Don't know; 4 = Somewhat effective; 5 = Very effective.

* **:red-background[opinion_seas_risk]** - Respondent's opinion about risk of getting sick with seasonal flu without vaccine.
1 = Very Low; 2 = Somewhat low; 3 = Don't know; 4 = Somewhat high; 5 = Very high.

* **:red-background[opinion_seas_sick_from_vacc]** - Respondent's worry of getting sick from taking seasonal flu vaccine.
1 = Not at all worried; 2 = Not very worried; 3 = Don't know; 4 = Somewhat worried; 5 = Very worried.

* **:red-background[age_group]** - Age group of respondent.

* **:red-background[education]** - Self-reported education level.

* **:red-background[ethnicity]** - Ethnicity of respondent.

* **:red-background[sex]** - Sex of respondent.

* **:red-background[income_poverty]** - Household annual income of respondent with respect to 2008 Census poverty thresholds.

* **:red-background[marital_status]** - Marital status of respondent.

* **:red-background[rent_or_own]** - Housing situation of respondent.

* **:red-background[employment_status]** - Employment status of respondent.

* **:red-background[hhs_geo_region]** - Respondent's residence using a 10-region geographic classification defined by the U.S. Dept. of Health and Human Services. Values are represented as short random character strings.

* **:red-background[census_msa]** - Respondent's residence within metropolitan statistical areas (MSA) as defined by the U.S. Census.

* **:red-background[household_adults]** - Number of other adults in household, top-coded to 3.

* **:red-background[household_children]** - Number of children in household, top-coded to 3.

* **:red-background[employment_industry]** - Type of industry respondent is employed in. Values are represented as short random character strings.

* **:red-background[employment_occupation]** - Type of occupation of respondent. Values are represented as short random character strings.
""");
st.markdown("""
##### **Data Source**
            
The data comes from the National 2009 H1N1 Flu Survey (NHFS) and is provided courtesy of the United States [National Center for Health Statistics](https://www.cdc.gov/nchs/index.htm) :link:.

In their own words:

The National 2009 H1N1 Flu Survey (NHFS) was sponsored by the National Center for Immunization and Respiratory Diseases (NCIRD) and conducted jointly by NCIRD and the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC). The NHFS was a list-assisted random-digit-dialing telephone survey of households, designed to monitor influenza immunization coverage in the 2009-10 season.

The target population for the NHFS was all persons 6 months or older living in the United States at the time of the interview. Data from the NHFS were used to produce timely estimates of vaccination coverage rates for both the monovalent pH1N1 and trivalent seasonal influenza vaccines.

The NHFS was conducted between October 2009 and June 2010. It was one-time survey designed specifically to monitor vaccination during the 2009-2010 flu season in response to the 2009 H1N1 pandemic. The CDC has other ongoing programs for annual phone surveys that continue to monitor seasonal flu vaccination.

---

##### **Data Restrictions**

The source dataset comes with the following data use restrictions:

* The Public Health Service Act (Section 308(d)) provides that the data collected by the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC), may be used only for the purpose of health statistical reporting and analysis.

* Any effort to determine the identity of any reported case is prohibited by this law.

* NCHS does all it can to ensure that the identity of data subjects cannot be disclosed. All direct identifiers, as well as any characteristics that might lead to identification, are omitted from the data files. Any intentional identification or disclosure of a person or establishment violates the assurances of confidentiality given to the providers of the information.

Therefore, users will:

* Use the data in these data files for statistical reporting and analysis only.

* Make no use of the identity of any person or establishment discovered inadvertently and advise the Director, NCHS, of any such discovery (1 (800) 232-4636).

* Not link these data files with individually identifiable data from other NCHS or non-NCHS data files.

* By using this data, you signify your agreement to comply with the above requirements.

<br>
""", unsafe_allow_html = True)'''

model_tab = '''st.markdown("""
The model used for classifying between "Vaccinated" and "Not vaccinated" is a multi-layer Perceptron classifier which optinises the log-loss function using LBFGS or stochastic gradient descent.
From [scikit-learn: 1.17.1. Multi-layer Perceptron](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) :link:

\"Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function $f: \mathrm R^m \\to \mathrm R^o$ by training on a dataset, 
where $m$ is the number of dimensions for input and $o$ is the number of dimensions for output. Given a set of features $X = x_1, x_2, \ldots, x_m$ and a target $y$, 
it can learn a non-linear function approximator for either classification or regression. It is different from logistic regression, 
in that between the input and the output layer, there can be one or more non-linear layers, called hidden layers. 

The leftmost layer, known as the input layer, consists of a set of neurons $\{x_i | x_1, x_2, \ldots, x_m\}$
representing the input features. Each neuron in the hidden layer transforms the values from the previous layer with a weighted linear summation
$w_1 x_1 + w_2 x_2 + \ldots + w_m x_m $, 
followed by a non-linear activation function $g(\cdot): \mathrm R \\to \mathrm R$ - like the hyperbolic tan function. 
The output layer receives the values from the last hidden layer and transforms them into output values.\"
""")'''