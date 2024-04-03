import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False
def parse_args():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="alibaba-fashion", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
    parser.add_argument('--load_rules', type=str2bool,default=True, help='whether to load outfit rules')
    parser.add_argument("--description", type=str, default="norule", help="description of the training")
    parser.add_argument('--which_rule', type=int, default=8, help='load which outfit rules')
    parser.add_argument('--rulescore_pos', type=float, default=0.001, help='coefficient of rulescore')
    parser.add_argument('--rulescore_neg', type=float, default=0.001, help='coefficient of rulescore')

    parser.add_argument('--score', type=str2bool, default=True, help='coefficient of rulescore')
    parser.add_argument('--score_change', type=str2bool, default=False, help='learning rate')

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs')  # using
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size') # using, original 1024
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=128, help='embedding size')   # using
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')  # using
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') # using

    parser.add_argument('--changelr2', type=str2bool, default=False, help='learning rate')
    parser.add_argument('--cosinelr', type=str2bool, default=False, help='cosine learning rate')
    parser.add_argument('--weightdecay', type=str2bool, default=False, help='optimizer weightdecay')
    parser.add_argument('--decayvalue', type=float, default=1e-4, help='decayvalue')
    parser.add_argument('--autolr', type=str2bool, default=False, help='adaptive learning rate')
    parser.add_argument('--autolr_loss', type=str2bool, default=False, help='adaptive learning rate')
    parser.add_argument('--pre_lr', type=str2bool, default=False, help='adaptive learning rate')
    parser.add_argument('--SGD', type=str2bool,default=False, help='SGD')

    parser.add_argument('--item', action='store_true', help='items')
    parser.add_argument('--outfit', action='store_true', help='outfits')
    parser.add_argument('--outfit_item', action='store_true', help='outfit_item')

    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=str2bool, default=False, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=str2bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--edge_threshold",type=int,default=64,help="edge threshold to filter knowledge graph")
    parser.add_argument("--adj_epoch", type=int, default=1, help="build adj matrix per_epoch")
    parser.add_argument("--in_channel", type=str, default="[64, 32]", help="input channels for gcn")
    parser.add_argument("--out_channel", type=str, default="[32, 64]", help="output channels for gcn")
    parser.add_argument( "--pretrain_s",type=bool,default=False,help="load pretrained sampler data or not",)
    parser.add_argument("--pretrain_r", type=bool, default=False, help="use pretrained model or not")
    parser.add_argument("--load_recommender", type=bool, default=False, help="load the recommender model with best performance")
    parser.add_argument("--load_sampler", type=bool, default=False, help="load the sampler model with best performance")
    parser.add_argument("--freeze_s",type=bool,default=False,help="freeze parameters of recommender or not",)
    parser.add_argument("--k_step", type=int, default=1, help="k step from current positive items")
    parser.add_argument("--num_sample", type=int, default=32, help="number fo samples from gcn")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma for reward accumulation")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
    parser.add_argument("--img_embedding_path", nargs="?", default="./data/alibaba-fashion/CLIP_embeddings_512_list.pickle", help="Input data path.")
    parser.add_argument("--txt_embedding_path", nargs="?", default="./data/alibaba-fashion/word_embeddings_384_list.pickle", help="Input data path.")

    # ===== added to run the code - zhou hang ===== #
    parser.add_argument('--temp_min', type=float, default=1e-1, help='temp_min')
    parser.add_argument('--temp_max', type=float, default=1e1, help='temp_max')
    parser.add_argument('--anneal_rate', type=float, default=9e-1, help='anneal_rate')
    parser.add_argument('--num_negs', type=int, default=1000, help='number of negatives candidates to select from')
    parser.add_argument('--tau', type=int, default=0.5, help='hyperparameter used in InfoNCE loss')
    parser.add_argument('--wcl', type=float, default=5e-2, help='weight for cl loss')
    # ===== added to run the code - zhou hang ===== #

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument("--rule_dict_path", nargs="?", default="./data/alibaba-fashion/rule_mapping_relations.npy", help="Input data path.")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")
    # ------------------------- experimental settings specific for testing ---------------------------------------------
    parser.add_argument("--rank", nargs="?", default="[20, 40, 60, 80, 100]", help="evaluate K list")       
    parser.add_argument("--flag_step", type=int, default=10, help="early stop steps")
    parser.add_argument('--test_step', type=int, default=1, help='test step')      # using

    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--num_neg', type=int, default=50)

    return parser.parse_args()
