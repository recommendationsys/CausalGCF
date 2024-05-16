# coding: utf-8
# author: lu yf
# create date: 2019-12-10 14:22

import torch
from torch.autograd import Variable


# Movielens dataset
class UserEmbeddingML(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingML, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']

        self.embedding_dim = config['embedding_dim']

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """

        :param user_fea:
        :return:
        """
        gender_idx = Variable(user_fea[:, 1], requires_grad=False)
        age_idx = Variable(user_fea[:, 2], requires_grad=False)
        occupation_idx = Variable(user_fea[:, 3], requires_grad=False)
        area_idx = Variable(user_fea[:, 4], requires_grad=False)
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)   # (1, 4*32)


class ItemEmbeddingML(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingML, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_director = config['num_director']
        self.num_year = config['num_year']
        self.embedding_dim = config['embedding_dim']

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        self.embedding_director = torch.nn.Embedding(
            num_embeddings=self.num_director,
            embedding_dim=self.embedding_dim
        )
        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """

        :param item_fea:
        :return:
        """
        rate_idx = Variable(item_fea[:, 1], requires_grad=False)
        genre_idx = Variable(item_fea[:, 2:27], requires_grad=False)
        director_idx = Variable(item_fea[:, 27], requires_grad=False)
        year_idx = Variable(item_fea[:, 28], requires_grad=False)
        rate_emb = self.embedding_rate(rate_idx)  # (1,32)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)  # (1,32)
        director_emb = self.embedding_director(director_idx)  # (1,32)
        year_emb = self.embedding_year(year_idx)  # (1,32)
        return torch.cat((rate_emb, genre_emb, director_emb, year_emb), 1)  # (1, 4*32)


# Yelp dataset
class UserEmbeddingYelp(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingYelp, self).__init__()
        self.num_fans = config['num_fans']
        self.num_avgrating = config['num_avgrating']
        self.embedding_dim = config['embedding_dim']

        self.embedding_fans = torch.nn.Embedding(
            num_embeddings=self.num_fans,
            embedding_dim=self.embedding_dim
        )

        self.embedding_avgrating = torch.nn.Embedding(
            num_embeddings=self.num_avgrating,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        fans_idx = Variable(user_fea[:, 0], requires_grad=False)  # [#sample]
        avgrating_idx = Variable(user_fea[:, 1], requires_grad=False)  # [#sample]
        fans_emb = self.embedding_fans(fans_idx)
        avgrating_emb = self.embedding_avgrating(avgrating_idx)
        return torch.cat((fans_emb, avgrating_emb), 1)   # (1, 1*32)


class ItemEmbeddingYelp(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingYelp, self).__init__()
        self.num_stars = config['num_stars']
        self.num_postalcode = config['num_postalcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_stars = torch.nn.Embedding(
            num_embeddings=self.num_stars,
            embedding_dim=self.embedding_dim,
        )

        self.embedding_postalcode = torch.nn.Embedding(
            num_embeddings=self.num_postalcode,
            embedding_dim=self.embedding_dim,
        )

    def forward(self, item_fea):
        stars_idx = Variable(item_fea[:, 0], requires_grad=False)
        postalcode_idx = Variable(item_fea[:, 1], requires_grad=False)

        stars_emb = self.embedding_stars(stars_idx)  # (1,32)
        postalcode_emb = self.embedding_postalcode(postalcode_idx)  # (1,32)
        return torch.cat((stars_emb, postalcode_emb), 1)


# DBook dataset
class UserEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingDB, self).__init__()
        self.num_location = config['num_location']  # 453
        self.embedding_dim = 32#config['embedding_dim']
        self.num_uid = config['num_uid']

        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )
        self.embedding_uid = torch.nn.Embedding(
            num_embeddings=self.num_uid,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """

        :param user_fea: tensor, shape = [#sample, #user_fea]
        :return:
        """
        user_idx = Variable(user_fea[:, 0], requires_grad=False)
        user_emb = self.embedding_uid(user_idx)
        location_idx = Variable(user_fea[:, 1], requires_grad=False)  # [#sample]
        location_emb = self.embedding_location(location_idx)
        return torch.cat((user_emb, location_emb), 1)  # (1, 1*32)


class ItemEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingDB, self).__init__()
        self.num_author = config['num_author']
        self.num_publisher = config['num_publisher']  # 1815
        self.num_year = config['num_year']  # 64
        self.num_iid = config['num_iid']
        self.embedding_dim = 32#32  # config['embedding_dim']  # 10802

        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher,
            embedding_dim=self.embedding_dim
        )
        self.embedding_author = torch.nn.Embedding(
            num_embeddings=self.num_author,
            embedding_dim=self.embedding_dim
        )
        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim
        )
        self.embedding_iid = torch.nn.Embedding(
            num_embeddings=self.num_iid,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """

        :param item_fea:
        :return:
        """
        item_idx = Variable(item_fea[:, 0], requires_grad=False)
        item_emb = self.embedding_iid(item_idx)
        year_idx = Variable(item_fea[:, 1], requires_grad=False)
        year_emb = self.embedding_year(year_idx)
        author_idx = Variable(item_fea[:, 2], requires_grad=False)
        author_emb = self.embedding_author(author_idx)
        publisher_idx = Variable(item_fea[:, 3], requires_grad=False)
        publisher_emb = self.embedding_publisher(publisher_idx)  # (1,32)
        return torch.cat((author_emb, publisher_emb), 1)  # (1, 1*32) item_emb, year_emb,

# book-crossing dataset
class UserEmbeddingBC(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingBC, self).__init__()
        self.num_location = 3454 #config['num_location']  # 453
        self.num_age = 240
        self.embedding_dim = 32#config['embedding_dim']
        self.num_uid = config['num_uid']

        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )
        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )
        self.embedding_uid = torch.nn.Embedding(
            num_embeddings=self.num_uid,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """

        :param user_fea: tensor, shape = [#sample, #user_fea]
        :return:
        """
        user_idx = Variable(user_fea[:, 0], requires_grad=False)
        user_emb = self.embedding_uid(user_idx)
        location_idx = Variable(user_fea[:, 1], requires_grad=False)  # [#sample]
        location_emb = self.embedding_location(location_idx)
        age_idx = Variable(user_fea[:, 2], requires_grad=False)  # [#sample]
        age_emb = self.embedding_age(age_idx)
        return torch.cat((user_emb, location_emb), 1)  # (1, 1*32)


class ItemEmbeddingBC(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingBC, self).__init__()
        self.num_author = 4211#config['num_author']
        self.num_publisher = 716#config['num_publisher']  # 1815
        self.num_year = 64#config['num_year']  # 64
        self.num_iid = config['num_iid']
        self.num_title = 11832
        self.embedding_dim = 32#32  # config['embedding_dim']  # 10802

        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher,
            embedding_dim=self.embedding_dim
        )
        self.embedding_author = torch.nn.Embedding(
            num_embeddings=self.num_author,
            embedding_dim=self.embedding_dim
        )
        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim
        )
        self.embedding_iid = torch.nn.Embedding(
            num_embeddings=self.num_iid,
            embedding_dim=self.embedding_dim
        )
        self.embedding_title = torch.nn.Embedding(
            num_embeddings=self.num_title,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """

        :param item_fea:
        :return:
        """
        item_idx = Variable(item_fea[:, 0], requires_grad=False)
        item_emb = self.embedding_iid(item_idx)
        year_idx = Variable(item_fea[:, 3], requires_grad=False)
        year_emb = self.embedding_year(year_idx)
        title_idx = Variable(item_fea[:, 1], requires_grad=False)
        title_emb = self.embedding_title(title_idx)
        author_idx = Variable(item_fea[:, 2], requires_grad=False)
        author_emb = self.embedding_author(author_idx)
        publisher_idx = Variable(item_fea[:, 4], requires_grad=False)
        publisher_emb = self.embedding_publisher(publisher_idx)  # (1,32)
        return torch.cat((author_emb, publisher_emb), 1)  # (1, 1*32) item_emb, year_emb,

# Amazon_Video_Games数据集
class UserEmbeddingVG(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingVG, self).__init__()
        self.num_location = 3454 #config['num_location']  # 453
        self.num_age = 240
        self.embedding_dim = 32#config['embedding_dim']
        self.num_uid = config['num_uid']

        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )
        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )
        self.embedding_uid = torch.nn.Embedding(
            num_embeddings=self.num_uid,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """

        :param user_fea: tensor, shape = [#sample, #user_fea]
        :return:
        """
        user_idx = Variable(user_fea[:, 0], requires_grad=False)
        user_emb = self.embedding_uid(user_idx)
        location_idx = Variable(user_fea[:, 1], requires_grad=False)  # [#sample]
        location_emb = self.embedding_location(location_idx)
        age_idx = Variable(user_fea[:, 2], requires_grad=False)  # [#sample]
        age_emb = self.embedding_age(age_idx)
        return torch.cat((user_emb, location_emb), 1)  # (1, 1*32)


class ItemEmbeddingVG(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingVG, self).__init__()
        self.num_title = 4252#config['num_author']
        self.num_caregoties = 461#config['num_publisher']  # 1815
        self.num_brand = 373#config['num_year']  # 64
        self.num_iid = config['num_iid']
        self.num_type = 112
        self.embedding_dim = 32#32  # config['embedding_dim']  # 10802

        self.embedding_caregoties = torch.nn.Embedding(
            num_embeddings=self.num_caregoties,
            embedding_dim=self.embedding_dim
        )
        self.embedding_brand = torch.nn.Embedding(
            num_embeddings=self.num_brand,
            embedding_dim=self.embedding_dim
        )
        self.embedding_type = torch.nn.Embedding(
            num_embeddings=self.num_type,
            embedding_dim=self.embedding_dim
        )
        self.embedding_iid = torch.nn.Embedding(
            num_embeddings=self.num_iid,
            embedding_dim=self.embedding_dim
        )
        self.embedding_title = torch.nn.Embedding(
            num_embeddings=self.num_title,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """

        :param item_fea:
        :return:
        """
        item_idx = Variable(item_fea[:, 0], requires_grad=False)
        item_emb = self.embedding_iid(item_idx)
        caregoties_idx = Variable(item_fea[:, 2], requires_grad=False)
        caregoties_emb = self.embedding_caregoties(caregoties_idx)
        title_idx = Variable(item_fea[:, 1], requires_grad=False)
        title_emb = self.embedding_title(title_idx)
        brand_idx = Variable(item_fea[:, 3], requires_grad=False)
        brand_emb = self.embedding_brand(brand_idx)
        type_idx = Variable(item_fea[:, 4], requires_grad=False)
        type_emb = self.embedding_type(type_idx)  # (1,32)
        return torch.cat((caregoties_emb, brand_emb), 1)  # (1, 1*32) item_emb, year_emb,