import torch
import torch.nn as nn
import torch.nn.functional as F


class Message(nn.Module):
    """
    One iteration of message passing
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.w2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.w3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)

    def forward(self, e, e_in, e_out):
        """
        :param e: Tensor (b, hidden_dim): edge embedding
        :param e_in: Tensor (b, hidden_dim): incoming edges embedding
        :param e_out: Tensor (b, hidden_dim): outgoing edges embedding
        """
        return F.relu(self.w1(e) + self.w2(e_in) + self.w3(e_out))


class GNNEncoder(nn.Module):
    """
    Graph encoder
    """
    def __init__(self, in_features, hidden_dim, num_iterations):
        super().__init__()
        self.num_iterations = num_iterations

        self.emb = nn.Linear(in_features=in_features, out_features=hidden_dim, bias=False)

        self.messages = nn.ModuleList([Message(hidden_dim=hidden_dim) for _ in range(num_iterations)])

        self.update = nn.GRU(input_size=2 * hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

    def forward(self, adj_matrix, edges):
        """
        :param adj_matrix: Tensor (V, V, in_features)
        :param edges: Tensor (E, 2)
        """
        x = self.emb(adj_matrix)  # (V, V, hidden_dim)

        for layer in range(self.num_iterations):
            m = self.message_passing(x, edges, layer)
            x, _ = self.update(torch.cat([m, x], dim=-1))

        # return torch.tanh(x)
        return x

    @staticmethod
    def _incoming_edges(e, adj_matrix):
        f_in = adj_matrix[:, e[0]].clone()
        f_in[e[1]] = 0
        N = torch.count_nonzero(f_in) / adj_matrix.shape[-1]
        if N == 0:
            N = 1
        return torch.sum(f_in, dim=0) / N

    def _outgoing_edges(self, e, adj_matrix):
        return self._incoming_edges(e.__reversed__(), adj_matrix)

    def message_passing(self, adj_matrix, edges, num_layer):
        h = adj_matrix.clone()
        for e in edges:
            h[e[0], e[1]] = self.messages[num_layer](adj_matrix[e[0], e[1]],               # self embedding
                                                     self._incoming_edges(e, adj_matrix),  # in embedding
                                                     self._outgoing_edges(e, adj_matrix))  # out embedding
        return h


class PathDecoder(nn.Module):
    """
    Path Decoder
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_state = torch.zeros((1, 1, hidden_dim))
        self.bidirectional = True
        self.path_decoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True,
                                   bidirectional=self.bidirectional)

    def forward(self, adj_matrix, paths, demand, hidden_state=None):
        h0 = hidden_state if hidden_state is not None else self.hidden_state
        if self.bidirectional:
            h0 = torch.cat([h0, h0], dim=0)
        x = []
        for i, p in enumerate(paths):
            path_enc = self.encode_path(adj_matrix, p, demand[:, [i], :].squeeze())
            x.append(self.path_decoder(path_enc, h0)[1])
        if self.bidirectional:
            return torch.mean(torch.stack(x, dim=0), dim=1).squeeze()
        return torch.stack(x, dim=0).squeeze()

    @staticmethod
    def encode_path(adj_matrix, path, demand):
        return torch.stack([torch.cat([adj_matrix[path[i], path[i + 1]], demand]) for i in range(len(path) - 1)], dim=0).unsqueeze(0)


class GNN(nn.Module):

    def __init__(self, in_features, hidden_dim, num_iterations, demand_hidden_dim=4):
        super().__init__()

        self.graph_encoder = GNNEncoder(in_features=in_features, hidden_dim=hidden_dim, num_iterations=num_iterations)
        self.path_decoder = PathDecoder(hidden_dim=hidden_dim + demand_hidden_dim)
        self.q = nn.Linear(in_features=hidden_dim + demand_hidden_dim, out_features=1, bias=True)
        self.demand_transform = nn.Linear(in_features=1, out_features=demand_hidden_dim, bias=False)

    def forward(self, adj_matrix, edges, paths, demands):
        """

        :param adj_matrix
        :param edges
        :param paths

        :return:
        """

        demand_emb = F.relu(self.demand_transform(demands.view((1, -1, 1))))  # (1, p, h_d)
        adj_emb = self.graph_encoder(adj_matrix, edges)  # (V, V, h)
        graph_emb = adj_emb.mean(dim=[0, 1], keepdims=True)  # (1, 1, h)
        graph_emb = torch.cat([graph_emb, torch.zeros_like(demand_emb[:, [0], :])], dim=-1)  # (1, 1, h+h_d)
        paths_emb = self.path_decoder(adj_matrix=adj_emb, paths=paths, hidden_state=graph_emb, demand=demand_emb) # (p, h+h_d)
        q_vals = F.softmax(self.q(paths_emb), dim=0)  # (p,1)

        return q_vals

GQN = GNN
