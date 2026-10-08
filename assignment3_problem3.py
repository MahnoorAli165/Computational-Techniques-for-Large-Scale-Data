 #!/usr/bin/env python3

from mrjob.job import MRJob

class MRJobTwitterFollows(MRJob):
    # The final (key,value) pairs returned by the class should be
    # 
    # yield ('most followed id', ???)
    # yield ('most followed', ???)
    # yield ('average followed', ???)
    # yield ('count follows no-one', ???)
    #
    # You will, of course, need to replace ??? with a suitable expression

    def mapper(self,key, value):
        user_id, follows = value.split(':', 1)
        follows = follows.strip()

        if follows == "":
            id_following_count = 0
        
        else:
            id_following_count = len(follows.split())

        yield "stats", (user_id, id_following_count, id_following_count == 0)

    def combiner(self, key, values):
        id_most_following = None
        following_max_count = -1
        following_all_users = 0
        number_users = 0
        count_zero_following = 0

        for value in values:
            user_id = value[0]
            following_count = value[1]
            following_zero = value[2]
            
            if following_count > following_max_count:
                id_most_following = user_id
                following_max_count = following_count
            
            following_all_users += following_count
            number_users += 1

            if following_zero:
                count_zero_following += 1
        
        yield "stats", (id_most_following, following_max_count, following_all_users, number_users, count_zero_following)


    def reducer(self,key, values):  
        #combiner

        id_most_following = None
        following_max_count = -1
        total_following_all_users = 0
        total_number_users = 0
        count_zero_following = 0

        for value in values:
            local_user_id = value[0]
            following_count = value[1]
            local_following_all_users = value[2]
            local_number_users = value[3]
            local_following_zero = value[4]
            
            if following_count > following_max_count:
                id_most_following = local_user_id
                following_max_count = following_count
            
            total_following_all_users += local_following_all_users
            total_number_users += local_number_users
            count_zero_following += local_following_zero


        average_following = total_following_all_users / total_number_users

        yield ("most followed id", id_most_following)
        yield ("most followed", following_max_count)
        yield ("average followed", average_following)
        yield ("count follows no-one", count_zero_following)

if __name__ == '__main__':
    MRJobTwitterFollows.run()

