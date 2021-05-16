def thruster(u,u_prev,off_start_time,fire_start_time,t):
    if(u>0):
        if u_prev == 0 and t-off_start_time<0.05:
            u=0
        else:
            u=1
            if u_prev == 0:
                fire_start_time = t
    if(u<=0):
        if u_prev == 1 and t-fire_start_time<0.05:
            u = 1
        else:
            u = 0
            if u_prev == 1:
                off_start_time = t
    return u,fire_start_time,off_start_time
