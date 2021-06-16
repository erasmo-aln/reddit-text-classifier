import re
import praw
import config

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from utils import constants, credentials


def connect_api():
    api_reddit = praw.Reddit(
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        user_agent=credentials.user_agent,
        username=credentials.username,
        password=credentials.password)

    return api_reddit


def char_count(post):
    post_length = len(re.sub(r'\W|\d', '', post.selftext))
    return post_length


def mask(post):
    mask = char_count(post) >= constants.MIN_LENGTH
    return mask


def load_data(subject_list):

    api_reddit = connect_api()

    data = list()
    labels = list()

    for label, subject in enumerate(subject_list):
        subreddit_data = api_reddit.subreddit(subject).new(limit=1000)

        post_list = [post.selftext for post in filter(mask, subreddit_data)]

        data.extend(post_list)
        labels.extend([label] * len(post_list))

        print(f'Number of r/{subject} posts: {len(post_list)}')

    return data, labels


def split_data(data, labels):

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=constants.TEST_SIZE,
        random_state=constants.RANDOM_STATE)

    return x_train, x_test, y_train, y_test


def preprocess_text(text):

    pattern = r'\W|\d|http.*\s+|www.*\s+'
    new_text = re.sub(pattern, ' ', text)

    return new_text


def preprocessing_pipeline():

    vectorizer = TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', min_df=constants.MIN_DOC_FREQ)
    decomposition = TruncatedSVD(n_components=constants.N_COMPONENTS, n_iter=constants.N_ITER)
    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]

    return pipeline


def build_model_list():

    knn_model = KNeighborsClassifier(n_neighbors=constants.N_NEIGHBORS)
    rf_model = RandomForestClassifier(random_state=constants.RANDOM_STATE)
    lr_model = LogisticRegressionCV(cv=constants.CV, random_state=constants.RANDOM_STATE)

    model_list = [('KNN', knn_model), ('RandomForest', rf_model), ('LogisticRegression', lr_model)]

    return model_list


def train_evaluation(model_list, pipeline, x_train, x_test, y_train, y_test):

    result_list = list()

    for model_name, model in model_list:
        pipe = Pipeline(pipeline + [(model_name, model)])

        print(f'Fitting training data with {model_name}...')
        pipe.fit(x_train, y_train)

        y_pred = pipe.predict(x_test)

        report = classification_report(y_test, y_pred)
        print(f'Classification Report:\n', report)

        result_list.append([model, {'model': model_name, 'predictions': y_pred, 'report': report}])

    return result_list
