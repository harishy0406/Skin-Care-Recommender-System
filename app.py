import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Import the Dataset
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Skin Care Recommender System", page_icon=":rose:", layout="wide",)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
                icons=["house", "stars", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    st.title(f"{selected} Product Recommender :sparkles:")
    st.write('---') 

    st.write(
    """
    ##### **The Skincare Product Recommendation Application is an implementation of Machine Learning that provides skincare product recommendations based on your skin type and concerns.**
    """
    )

    
    #displaying a local video file

    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time = 1) #displaying the video 
    
    st.write(' ') 
    st.write(' ')
    st.write(
    """
    ##### You will receive skincare product recommendations from various cosmetic brands with a total of 1200+ products tailored to your skin's needs. 
    ##### There are 5 categories of skincare products for 5 different skin types, as well as concerns and benefits you want to achieve from the products. This recommendation application is just a system that provides recommendations based on the data you input, not scientific consultation.
    ##### Please select the *Get Recommendation* page to start receiving recommendations, or select the *Skin Care 101* page to view tips and tricks about skincare.
    """
    )

    st.write(
        """
        **Good Luck :) !**
        """
    )

    
    
    st.info('Credit: Created by Chandini')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    
    st.write(
    """
    ##### **To get recommendations, please enter your skin type, concerns, and desired benefits to receive the right skincare product recommendations**
    """
    )
 
    
    st.write('---') 

    first,last = st.columns(2)

    # Choose a product product type category
    # pt = product type
    category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    # st = skin type
    skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'] )
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    
    prob = st.multiselect(label='Skin Problems: ', options= ['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Loose Skin'])


    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    
    selected_options = st.multiselect('Notable Effects : ',opsi_ne)
    
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    
    product = st.selectbox(label='Product Recommended For You', options=sorted(opsi_pn))

    

    
    tf = TfidfVectorizer()

    
    tf.fit(skincare['notable_effects']) 

    
    tf.get_feature_names_out()

    
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

    
    shape = tfidf_matrix.shape

    
    tfidf_matrix.todense()

    
    pd.DataFrame(
        tfidf_matrix.todense(), 
        columns=tf.get_feature_names_out(),
        index=skincare.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    
    cosine_sim = cosine_similarity(tfidf_matrix) 

    
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    
    cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

    
    def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):

       
        index = similarity_data.loc[:,nama_produk].to_numpy().argpartition(range(-1, -k, -1))

        
        closest = similarity_data.columns[index[-1:-(k+2):-1]]

        
        closest = closest.drop(nama_produk, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df

    
    model_run = st.button('Find Other Product Recommendations!')
 
    if model_run:
        st.write('Here are other similar product recommendations based on your preferences')
        st.write(skincare_recommendations(product))
    
    
if selected == "Skin Care 101":
    st.title(f"Take a Look at {selected}")
    st.write('---') 

    st.write(
    """
    ##### **Here are some tips and tricks you can follow to maximize the use of skin care products**
    """)
 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='Skin Care 101')
    

    
    st.write(
    """
    ### **1. Facial Wash**
    """)
    st.write(
    """
    **- Use a facial wash product that has been recommended or one that suits you**
    """)
    st.write(
        """
        **- Wash your face a maximum of 2 times a day, in the morning and before bed. Washing your face too often will remove the skin's natural oils. If you have dry skin, it's okay to only use water in the morning**
        """)
    st.write(
        """
        **- Do not rub your face harshly as it can remove the skin's natural protective layer**
        """)
    st.write(
        """
        **- The best way to cleanse your skin is by using your fingertips for 30-60 seconds with circular motions and gentle massage**
        """)
        
    st.write(
        """
        ### **2. Toner**
        """)
    st.write(
        """
        **- Use a toner that has been recommended or one that suits you**
        """)
    st.write(
        """
        **- Apply toner to a cotton pad and gently swipe it across your face. For better results, use two layers of toner: first with a cotton pad and the last layer with your hands to help it absorb better**
        """)
    st.write(
        """
        **- Use toner after cleansing your face**
        """)
    st.write(
        """
        **- If you have sensitive skin, try to avoid skin care products containing fragrance**
        """)
        
    st.write(
        """
        ### **3. Serum**
        """)
    st.write(
        """
        **- Use a serum that has been recommended or one that suits you for better results**
        """)
    st.write(
        """
        **- Apply serum after your face is completely clean to ensure maximum absorption**
        """)
    st.write(
        """
        **- Use serum in the morning and at night before bed**
        """)
    st.write(
        """
        **- Choose a serum according to your needs, such as for acne scars, dark spots, anti-aging, or other benefits**
        """)
    st.write(
        """
        **- To help serum absorb better, pour it onto your palms and gently tap it onto your face, then wait for it to absorb**
        """)
        
    st.write(
        """
        ### **4. Moisturizer**
        """)
    st.write(
        """
        **- Use a moisturizer that has been recommended or one that suits you for better results**
        """)
    st.write(
        """
        **- Moisturizer is a must-have product as it locks in the moisture and nutrients from the serum**
        """)
    st.write(
        """
        **- For better results, use different moisturizers for the morning and night. Day moisturizers usually contain sunscreen and vitamins to protect the skin from UV rays and pollution, while night moisturizers contain active ingredients that help skin regeneration during sleep**
        """)
    st.write(
        """
        **- Give a 2-3 minute gap between using serum and moisturizer to ensure the serum has absorbed into your skin**
        """)
        
    st.write(
        """
        ### **5. Sunscreen**
        """)
    st.write(
        """
        **- Use sunscreen that has been recommended or one that suits you for better results**
        """)
    st.write(
        """
        **- Sunscreen is the key product in any skincare routine as it protects your skin from harmful UVA and UVB rays, even from blue light. All other skin care products are less effective if you don't protect your skin**
        """)
    st.write(
        """
        **- Apply sunscreen about the length of your index and middle fingers for maximum protection**
        """)
    st.write(
        """
        **- Reapply sunscreen every 2-3 hours or as needed**
        """)
    st.write(
        """
        **- Always use sunscreen, even indoors, as UV rays can penetrate windows after 10 AM and even on cloudy days**
        """)
        
    st.write(
        """
        ### **6. Avoid Frequently Changing Skin Care Products**
        """)
    st.write(
        """
        **Frequently switching skin care products can cause stress to your skin as it tries to adapt to the new ingredients. As a result, the benefits will not be fully realized. Instead, use a product consistently for months to see its effects**
        """)
        
    st.write(
        """
        ### **7. Consistency**
        """)
    st.write(
        """
        **The key to skin care is consistency. Be diligent and persistent in using your skin care products, as the results are not instant**
        """)
    st.write(
        """
        ### **8. Your Face is an Asset**
        """)
    st.write(
        """
        **Every human face is a blessing from the Creator. Take good care of it as a form of gratitude. Choose products and treatments that are suited to your skin's needs. Using skin care products early is like making an investment for your future**
        """)

        
    
    
