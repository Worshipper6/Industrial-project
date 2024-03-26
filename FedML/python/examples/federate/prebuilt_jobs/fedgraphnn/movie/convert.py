import csv
f1 = open('/Users/dujiaxing/FedML/python/examples/federate/prebuilt_jobs/fedgraphnn/recsys_subgraph_link_pred/data/ml-small/ratings.csv', 'r')
f2 = open('/Users/dujiaxing/FedML/python/examples/federate/prebuilt_jobs/fedgraphnn/recsys_subgraph_link_pred/data/ml-small/user.dict', 'w')
f3 = open('/Users/dujiaxing/FedML/python/examples/federate/prebuilt_jobs/fedgraphnn/recsys_subgraph_link_pred/data/ml-small/item.dict', 'w')
f1.readline()

user_list = []
movie_list = []
for line in f1:
    # remove leading and trailing whitespace
    line = line.strip()
    user, movie, rate, t = line.split(',')
    user_list.append(user)
    movie_list.append(movie)
user_list = list(set(user_list))
movie_list = list(set(movie_list))

# user.dict
i = 0
for user in user_list:
    f2.write('%s\t%s\n' % (user, i))
    i += 1

# item.dict
for item in movie_list:
    f3.write('%s\t%s\n' % (item, i))
    i += 1

f1.close()
f2.close()
f3.close()