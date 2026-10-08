 #!/usr/bin/env python3

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJobTwitterFollowers(MRJob):
    # The final (key,value) pairs returned by the class should be
    # 
    # yield ('most followers id', ???)
    # yield ('most followers', ???)
    # yield ('average followers', ???)
    # yield ('count no followers', ???)
    #
    # You will, of course, need to replace ??? with a suitable expression

    # as from the hint in the assigmnent we might need to implement this in steps, therefore MRStep is imported.

    def steps(self):

        # first stepcomputes user_id and follower_count
        return [
            MRStep(mapper=self.mapper_followers, combiner=self.combiner_follower_count, reducer=self.reducer_follower_count),
            MRStep(mapper=self.mapper_stats, combiner=self.combiner_stats, reducer=self.reducer_stats)
        
        ]

    def mapper_followers(self,key, value):
        user_id, follows = value.split(':', 1)
        follows = follows.strip()

        yield user_id, 0

        if follows != "":
           for followed_user in follows.split():
                yield followed_user, 1
        

    def combiner_follower_count(self, user_id, values):
        follower_count = sum(values)
        yield user_id, follower_count

    def reducer_follower_count(self,user_id, values):
        follower_count = sum(values)
        yield user_id, follower_count
        
#------------------Step 2------------------#
    def mapper_stats(self, user_id, follower_count):
        yield "stats", (user_id, follower_count, follower_count, 1, follower_count == 0)

    def combiner_stats(self, key, values):
        id_most_followers = None
        followers_max_count = -1
        followers_all_users = 0
        number_users = 0
        count_zero_followers = 0

        for value in values:
            user_id = value[0]
            local_follower_count = value[1]
            local_followers = value[2]
            local_users = value[3]
            local_followers_zero = value[4]

            if local_follower_count > followers_max_count:
                id_most_followers = user_id
                followers_max_count = local_follower_count
            
            followers_all_users += local_followers
            number_users += local_users
            count_zero_followers += local_followers_zero

        
        yield "stats", (id_most_followers, followers_max_count, followers_all_users, number_users, count_zero_followers)

    def reducer_stats(self, key, values):
        id_most_followers = None
        followers_max_count = -1
        total_followers_all_users = 0
        total_number_users = 0
        count_zero_followers = 0

        for value in values:
            user_id = value[0]
            local_follower_count_max = value[1]
            local_followers = value[2]
            local_users = value[3]
            local_followers_zero = value[4]
            if local_follower_count_max > followers_max_count:
                id_most_followers = user_id
                followers_max_count = local_follower_count_max

            total_followers_all_users += local_followers
            total_number_users += local_users
            count_zero_followers += local_followers_zero

        average_followers = total_followers_all_users / total_number_users 

        yield "most followers id", id_most_followers
        yield "most followers", followers_max_count     
        yield "average followers", average_followers
        yield "count no followers", count_zero_followers

            




if __name__ == '__main__':
    MRJobTwitterFollowers.run()

