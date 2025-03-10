import torch
import torch.nn as nn
import math


class ViT_model(nn.Module):
    """
    Vision Transformer模型主类，包含三个主要部分：嵌入层、Transformer编码器、MLP_Head
    """
    def __init__(self):
        super(ViT_model, self).__init__()
        # 分为embedding layer, transformer encoder, MLP_head三部分
        self.embeddings = ViTEmbeddings()
        self.trans_encoder = ViTEncoder()
        self.mlp_head = MLP_head()
        self.cls = nn.Linear(768, 2)    # 二分类

    def forward(self, pixel_value):
        hidden_states = self.embeddings(pixel_value)    # 获取嵌入向量
        hidden_states = self.trans_encoder(hidden_states)
        # poolout指cls，last_hidden指cls+all_patches,  cls:[b, 768]  last_hidden:[b, 197, 768]
        poolout, last_hidden = self.mlp_head(hidden_states)
        # 使用poolout做分类
        logtic = self.cls(poolout)

        return logtic


# 图像嵌入层
class ViTEmbeddings(nn.Module):
    '''
    1.将图片打散成patches
    2.cls token 在每个batch都要拼接patch
    3.pos_emb + patches
    '''
    def __init__(self):
        super(ViTEmbeddings, self).__init__()
        image_size = 224    # 每张图片大小
        patch_size = 16     # 每个patch大小
        num_channels = 3     # 每个patch的通道数
        hidden_size = 768   # 每个图片表征的维度，16*16*3=768
        num_patches = (image_size // patch_size) ** 2       # 224//16=14,14*14=196    196个token

        # 1.[b, 3, 224, 224] -> [b, 196, 768]
        self.image2patch = Image2Patch(image_size, patch_size, num_channels, hidden_size)
        # 2.扩展 [1,1,768] -> [b,1,768], cls_token用于分类任务
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        # 3.pos_encoding [b, 197, 768], 位置编码，用于添加位置信息
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches+1, hidden_size))

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]

        # [b, 3, 224, 224] -> [b, 196, 768]
        patch_emb = self.image2patch(hidden_states)
        # 扩展 [1,1,768] -> [b,1,768]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        embs = torch.cat((cls_tokens, patch_emb), dim=1)

        # 将patch_emb和位置编码相加 [b, 197, 768] + [b, 197, 768] -> [b, 197, 768]
        embs = embs+self.pos_encoding
        return embs


# 将图像分割为patch并转换为嵌入向量
class Image2Patch(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super(Image2Patch, self).__init__()
        self.projection = nn.Conv2d(in_channels=num_channels, out_channels=hidden_size, kernel_size=(patch_size, patch_size), stride=patch_size)

    def forward(self, hidden_states):
        # hidden_states: [b, 3, 224, 224] -> [b, 768, 14, 14]
        x = self.projection(hidden_states)
        # 展平    [b, 768, 14, 14] -> [b, 768, 196] -> [b, 196, 768]
        x = x.flatten(2).transpose(-1, -2)
        return x


# Transformer编码器，包含多个Transformer层
class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.layer = nn.ModuleList([TransLayer() for _ in range(6)])

    def forward(self, hidden_states):
        # [b, 197, 768]
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


# 单个Transformer层，包含多头自注意力机制和前馈网络
class TransLayer(nn.Module):
    def __init__(self):
        super(TransLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(768)    # 第一个层归一化
        self.multi_head = Self_attention()      # 多头自注意力机制
        self.dropout = nn.Dropout(0.1)          # Dropout层
        self.layer_norm2 = nn.LayerNorm(768)    # 第二个层归一化
        self.mlp_block = Mlp_block()            # 前馈网络

    def forward(self, hidden_states):
        layer_norm1_out = self.layer_norm1(hidden_states)       # [b, 197, 768]
        multi_head_out = self.multi_head(layer_norm1_out)       # [b, 197, 768]

        hidden_states = hidden_states + multi_head_out          # [b, 197, 768] + [b, 197, 768] = [b, 197, 768]

        layer_norm2_out = self.layer_norm2(hidden_states)       # [b, 197, 768]
        mlp_block_out = self.mlp_block(layer_norm2_out)         # [b, 197, 768]

        hidden_states = hidden_states + mlp_block_out           # [b, 197, 768] + [b, 197, 768] = [b, 197, 768]

        return hidden_states


class Self_attention(nn.Module):
    """
    x:[b, 197, 768]
    1. Q, K, V -> [b, 197, 768]
    2. transpose_for_scores -> [b, 4, 197, 192]
    3. clulate_score  QK.T / d_k**-0.5  ->  softmax  -> V  ->  [b, 4, 197, 192]
    4. head_cat  [b, 197, 768]
    """
    def __init__(self):
        super(Self_attention, self).__init__()
        self.num_atten_head = 4         # 头数
        self.num_atten_head_size = 768//4       # 每个头的维度
        self.all_hidden_size = self.num_atten_head * self.num_atten_head_size
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(768, 768)

    # 拆分多头
    def transpose_for_scores(self, x):
        # [b, 197, 768] -> [b, 4, 197, 192]     x.size()[:-1] -> [b, 197] + [4, 192] -> [b, 197, 4, 192]
        new_x_shape = x.size()[:-1] + (self.num_atten_head, self.num_atten_head_size)
        # [b, 197, 768] -> [b, 4, 197, 192]   x.view() -> [b, 197, 4, 192]    x.permute() -> [b, 4, 197, 192]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # [b, 197, 768]
        mixed_query = self.query(hidden_states)  # [b, 197, 768]
        mixed_key = self.key(hidden_states)    # [b, 197, 768]
        mixed_value = self.value(hidden_states)  # [b, 197, 768]

        # [b, 4, 197, 192]
        query_layer = self.transpose_for_scores(mixed_query)
        key_layer = self.transpose_for_scores(mixed_key)
        value_layer = self.transpose_for_scores(mixed_value)

        # 计算注意力分数
        # QK.T / d_k**-0.5  ->  softmax  -> V  ->  [b, 4, 197, 192]
        # [b, 4, 197, 192]@[b, 4, 192, 197]->[b,4,197,197]
        atten_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.num_atten_head_size)

        # softmax  [b,4,197,197]
        atten_pro = self.softmax(atten_scores)

        # score V  [b,4,197,197]@[b, 4, 197, 192]->[b, 4, 197, 192]
        context_layer = torch.matmul(atten_pro, value_layer)

        # [b, 4, 197, 192] -> [b, 197, 4, 192] -> [b, 197, 768]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer = context_layer.size()[:-2] + (self.all_hidden_size,)
        context_layer = context_layer.view(*new_context_layer)

        # [b, 197, 768]
        atten_out = self.fc(context_layer)
        atten_out = self.dropout(atten_out)

        return atten_out


# 前馈网络
class Mlp_block(nn.Module):
    def __init__(self):
        super(Mlp_block, self).__init__()
        self.fc1 = nn.Linear(768, 3072)     # 第一个全连接层
        self.gelu = nn.GELU()                                    # GELU激活函数
        self.fc2 = nn.Linear(3072, 768)     # 第二个全连接层
        self.dropout = nn.Dropout(0.1)                           # Dropout层

    def forward(self, hidden_states):
        # [b, 197, 768]
        x = self.fc1(hidden_states)
        # [b, 197, 3072]
        x = self.gelu(x)
        x = self.dropout(x)
        # [b, 197, 3072]
        x = self.fc2(x)
        # [b, 197, 768]
        x = self.dropout(x)
        # [b, 197, 768]
        return x


# 多层感知机头部，用于处理Transformer编码器的输出
class MLP_head(nn.Module):
    def __init__(self):
        super(MLP_head, self).__init__()
        self.linear1 = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(768, 768)

    def forward(self, hidden_states):
        cls_token = hidden_states[:, 0]
        poolout = self.linear1(cls_token)
        poolout = self.activation(poolout)
        poolout = self.linear2(poolout)

        # poolout:[b, 768], [b, 197, 768]
        return poolout, hidden_states


if __name__ == '__main__':
    tensor = torch.randn(10, 3, 224, 224)       # 测试模型
    vit = ViT_model()
    out = vit(tensor)
    print(out.shape)