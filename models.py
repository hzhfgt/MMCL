import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = pyg.nn.GATConv(in_channel, out_channel)
        self.conv2 = pyg.nn.GATConv(in_channel, out_channel)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)
        # x = self.conv2(x, edge_indices)
        x = self.dropout(x)
        x = F.normalize(x)
        return x



class MM_model(nn.Module):
    def __init__(self, args, dim_ent, dim_img, dim_txt, n_nodes):
        super(MM_model, self).__init__()
        self.args = args
        self.ent_mapping = nn.Linear(dim_ent, self.args.dim)
        self.img_mapping = nn.Linear(dim_img, self.args.dim)
        self.txt_mapping = nn.Linear(dim_txt, self.args.dim)
        self.entity_graphconv = GraphConv(self.args.dim, self.args.dim)
        self.image_graphconv = GraphConv(self.args.dim, self.args.dim)
        self.txt_graphconv = GraphConv(self.args.dim, self.args.dim)

        self.fusion_map = nn.Linear(3*dim_ent, 3*dim_ent)
        self.fusion = nn.Linear(3*dim_ent, self.args.dim)

        self.entity_feat = nn.Parameter(
            torch.FloatTensor(n_nodes, self.args.dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.entity_feat)

    def forward(self, entity_feat_1, image_feat, text_feat, graph_edges):
        '''
        input
        entity_feat: (num_graph_nodes, dim_ent)
        image_feat: (num_graph_nodes, dim_img)
        text_feat: (num_graph_nodes, dim_txt)
        graph_edges: (2, num_of_edges)
        output
        mm_feat: (num_graph_nodes, dim_mm)
        '''
        # entity_feat = self.ent_mapping(entity_feat)
        entity_feat = self.entity_feat
        image_feat = self.img_mapping(image_feat)
        text_feat = self.txt_mapping(text_feat)

        entity_feat = self.entity_graphconv(entity_feat, graph_edges)
        image_feat = self.image_graphconv(image_feat, graph_edges)
        text_feat = self.txt_graphconv(text_feat, graph_edges)

        # mm_feat = (entity_feat + image_feat + text_feat)/3
        mm_feat = torch.cat((entity_feat, image_feat, text_feat), 1)
        # mm_feat = self.fusion_map(mm_feat)
        mm_feat = self.fusion(mm_feat)
        # mm_feat = entity_feat

        return entity_feat, image_feat, text_feat, mm_feat
