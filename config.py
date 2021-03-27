import torch
num_user = 200000
num_item = 136736
vec_dim = 64
user_item_vecs="user_item_vectors64_train_0_likunlik.npz"  # from BPR
train_json_file = "alluserinfo_train_0.json"
test_json_file = "alluserinfo_test_0.json"
model_root = "models"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


eval_users = range(10000)  # just an example or None(entire users), users to be evaluated
candidate_size=136736  # The total # of items. In evaluation, recommend topN items from this list, e.g., candidate=["like"]+["dislike"]+["weak_neg"])
topN = 1000 # recommend top items, must be a positive integer

batch_size=1024*8
num_positive = 4
num_negative = 4
epoch = 5000
verbose = 1
lr_embedding = 1e-5  # make it small or 0, because user and item vector are pre-learned by BPR
lr = 1e-3
beta1 = 0.9
beta2 = 0.999

# below is settings related to content, you need to prepare your own data
item_genre_file = "..\\dataset\\genre-hierarchy.txt"
item_content_file = "..\\dataset\\song-attributes.txt"
num_genre_type = 216 #0 unknown, valid genre: 1~215
content_vec_dim = 64
#num_same_tree_genre = 12  # averagely 12, but varyies individually
num_negative_genre = 12
genreNet_epoch = 4000
genre_lr = 1e-3
genre_batchsize = 512

genre_encoder = "genreEncoder_vec64_0.85train_4000epoch.pt"
lr_genre_encoder = 0.0