### 异常情况

- **cannot assign requested address**

  ![image-20220620103752066](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220620103752066.png)

  - **异常解决**

    ![image-20220620105512538](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20220620105512538.png)

    云服务器中会出现两个IP地址，一个是私有的（业务使用），一个是弹性的（管理使用），由于将弹性IP地址用作接口host导致出现了上述异常情况，关于机器使用的IP地址情况，可以通过使用如下命令查看：

    ```shell
    # 方法一：查看网卡信息
    ifconfig
    
    
    eno1      Link encap:以太网  硬件地址 2c:fd:a1:c6:0b:4a
              inet 地址:192.168.51.58  广播:192.168.51.255  掩码:255.255.255.0
              inet6 地址: fe80::c251:5b7b:be7b:f7cf/64 Scope:Link
              UP BROADCAST RUNNING MULTICAST  MTU:1500  跃点数:1
              接收数据包:278878876 错误:0 丢弃:1522184 过载:0 帧数:0
              发送数据包:261485315 错误:0 丢弃:0 过载:0 载波:0
              碰撞:0 发送队列长度:1000
              接收字节:97524353198 (97.5 GB)  发送字节:105326017540 (105.3 GB)
              中断:20 Memory:fb600000-fb620000
              
    
    # 方法二：
    cat /etc/hosts
    
    # 查看所有的host
    	
    ```

  - 情形二：端口被占用，有程序在运行，此时如果可以选择其他端口启动服务

    ```shell
    lsof -i:7893
    
    # 若无输出表示未被占用
    ```

    

  - 情形三：选用的端口不在开启的端口范围内，查看开启的端口范围：

    ```shell
    # 查看当前linux系统的可分配端口
    cat /proc/sys/net/ipv4/ip_local_port_range
    
    # 修改端口范围,1000到65534可供用户程序使用，1000以下为系统保留端口
    vim /etc/sysctl.conf
    
    # 添加如下内容：
    net.ipv4.ip_local_port_range = 1000 65534
    
    # 执行
    sysctl -p
    
    # 再次查看端口范围可以发现范围发生变化。
    ```

     配置tcp端口的重用配置，提高端口的回收效率

    ```shell
    vim /etc/sysctl.conf
    ```

    添加如下内容：

    ```shell
    #TCP connection recovery
    
    net.ipv4.tcp_max_tw_buckets = 6000000
    net.ipv4.tcp_tw_reuse = 1
    net.ipv4.tcp_tw_recycle = 1
    net.ipv4.tcp_fin_timeout = 10
    net.ipv4.route.max_size = 5242880
    net.ipv4.ip_forward = 1
    net.ipv4.tcp_timestamps = 1
    
    ```

- 

