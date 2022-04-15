from matplotlib.pyplot import text
import streamlit as st
import pandas as pd

data_dir = '../data/hateful_memes'

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache
def load_data(split='all'):
    df = pd.read_csv(f'{data_dir}/info.csv')
    df['id'] = df['id'].astype('str')
    df.index = df['id']
    df.index.name = None
    if split != 'all':
        df = df[df['split']==split]
    return df

def check_filter(hateful_meme_idx, filter_criterion):
    global df, df_non_hateful

    if filter_criterion == '1 benign confounder - img':
        text_idx = df.at[hateful_meme_idx, 'pseudo_text_idx']
        if text_idx in df_non_hateful['pseudo_text_idx'].values.tolist():
            return True
        else:
            return False
    elif filter_criterion == '1 benign confounder - text':
        img_idx = df.at[hateful_meme_idx, 'pseudo_img_idx']
        if img_idx in df_non_hateful['pseudo_img_idx'].values.tolist():
            return True
        else:
            return False
    elif filter_criterion == '1 benign confounder - img & text':
        text_idx = df.at[hateful_meme_idx, 'pseudo_text_idx']
        img_idx = df.at[hateful_meme_idx, 'pseudo_img_idx']
        if text_idx in df_non_hateful['pseudo_text_idx'].values.tolist() and img_idx in df_non_hateful['pseudo_img_idx'].values.tolist():
            return True
        else:
            return False
    elif filter_criterion == '1 benign confounder - img | text':
        text_idx = df.at[hateful_meme_idx, 'pseudo_text_idx']
        img_idx = df.at[hateful_meme_idx, 'pseudo_img_idx']
        if text_idx in df_non_hateful['pseudo_text_idx'].values.tolist() or img_idx in df_non_hateful['pseudo_img_idx'].values.tolist():
            return True
        else:
            return False
    elif filter_criterion == 'none':
        text_idx = df.at[hateful_meme_idx, 'pseudo_text_idx']
        img_idx = df.at[hateful_meme_idx, 'pseudo_img_idx']
        if text_idx in df_non_hateful['pseudo_text_idx'].values.tolist() or img_idx in df_non_hateful['pseudo_img_idx'].values.tolist():
            return False
        else:
            return True
    else:
        raise ValueError()

@st.cache
def filter_hateful_memes(hateful_meme_idxs, filter_criterion):
    hateful_meme_idxs_filtered = [hateful_meme_idx for hateful_meme_idx in hateful_meme_idxs if check_filter(hateful_meme_idx, filter_criterion)]
    return hateful_meme_idxs_filtered

# select the data split
splits = ['all', 'train', 'dev_seen', 'test_seen']
split = st.sidebar.selectbox("Select the data split", splits)
df = load_data(split)
label_counts = df['label'].value_counts()
st.sidebar.write(f"Non-hateful + Hateful = {label_counts[0]} + {label_counts[1]} = {len(df)}")

# filter the hateful memes
filters = ['<no filter>', '1 benign confounder - img', '1 benign confounder - text', '1 benign confounder - img & text', '1 benign confounder - img | text', 'none']
filter_criterion = st.sidebar.selectbox("Filter the hateful memes that have atleast", filters)

hateful_meme_idxs = sorted(df[df['label']==1]['id'].values.tolist(), key=int)
if filter_criterion == '<no filter>':
    hateful_meme_idxs_filtered = hateful_meme_idxs
else:
    df_non_hateful = df[df['label']==0]
    df_hateful = df[df['label']==1]
    hateful_meme_idxs_filtered = filter_hateful_memes(hateful_meme_idxs, filter_criterion)
st.sidebar.write(f'{len(hateful_meme_idxs)} -> {len(hateful_meme_idxs_filtered)}')

# select the hateful meme 
hateful_meme_idx = st.sidebar.select_slider("Select the hateful meme", options=hateful_meme_idxs_filtered)

# display the hateful meme
img_idx, text_idx = df.loc[hateful_meme_idx, ['pseudo_img_idx', 'pseudo_text_idx']]
caption = f'meme={hateful_meme_idx}; img={img_idx}; text={text_idx}'
meme_path = f"{data_dir}/{df.loc[hateful_meme_idx, 'img']}"
st.sidebar.image(meme_path, caption, width=400)
try:
   #st.sidebar.image(meme_path.replace('memes/img', 'memes_f1'), caption+"; F1", width=400) 
   st.sidebar.image(meme_path.replace('memes/img', 'memes_f2'), caption+"; F2", width=400) 
except:
    pass
# filter the non hatefull data
df = df[df['label']==0]

# benign confounders - images
st.write("### Benign Confounders - Images")
meme_idxs_matching_text = df[df['pseudo_text_idx']==text_idx]['id'].values.tolist()
#st.write(meme_idxs_matching_text)
meme_paths_matching_text = [f"{data_dir}/{df.loc[meme_idx, 'img']}" for meme_idx in meme_idxs_matching_text]
for  meme_path_matching_text, meme_idx_matching_text in  zip(meme_paths_matching_text, meme_idxs_matching_text):
    st.image(meme_path_matching_text, meme_idx_matching_text, width=400)
    try:
        #st.image(meme_path_matching_text.replace('memes/img', 'memes_f1'), meme_idx_matching_text, width=400) 
        st.image(meme_path_matching_text.replace('memes/img', 'memes_f2'),  meme_idx_matching_text, width=400) 
    except:
        pass
if len(meme_idxs_matching_text) == 0:
    st.write("*** No Matching Found ***")

st.write("-"*10)

# benign confounders - texts
st.write("### Benign Confounders - Texts")
meme_idxs_matching_img = df[df['pseudo_img_idx']==img_idx]['id'].values.tolist()
#st.write(meme_idxs_matching_img)
meme_paths_matching_img = [f"{data_dir}/{df.loc[meme_idx, 'img']}" for meme_idx in meme_idxs_matching_img]
for meme_path_matching_img, meme_idx_matching_img in  zip(meme_paths_matching_img, meme_idxs_matching_img):
    st.image(meme_paths_matching_img, meme_idxs_matching_img, width=400)
    try:
        #st.image(meme_path_matching_img.replace('memes/img', 'memes_f1'), meme_idx_matching_img, width=400) 
        st.image(meme_path_matching_img.replace('memes/img', 'memes_f2'),  meme_idx_matching_img, width=400) 
    except:
        pass
if len(meme_idxs_matching_img) == 0:
    st.write("*** No Matching Found ***")