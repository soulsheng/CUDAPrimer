参考教程：《深入浅出谈CUDA》 网页和Word文档详见：

http://www.cppblog.com/hktk/category/11855.html

http://sdrv.ms/SkhzZ2  


CUDAPrimer 0.469, 2012-09-24 ----------------------- 
- 两种查看每个线程的寄存器使用量的方法：方法一：开启编译选项： --ptxas-options=-v，查看编译日志： $(IntDir)/工程名字.log；方法二：CUDA Visual Profiler 分析exe，通过exe目录下csv文件，regperthread字段代表寄存器使用量。
- 获取最优化线程数目配置，通过由cuda sdk提供的计算工具CUDA_Occupancy_Calculator.xls 。

CUDAPrimer 0.468, 2012-09-21 ----------------------- 
- 不同局部变量，占用多少寄存器？性能有什么影响？总结，最多使用6个寄存器，变量最多定义5个，超过后将采用局部变量，耗时增至3、10、20倍，效率陡降到1/3、1/10、1/20

CUDAPrimer 0.467, 2012-09-21 ----------------------- 
- 测量cpu耗时。不同问题规模，cuda加速性能发挥不同，问题规模由数据元素个数决定。- 在gts250机器上，问题规模达到2M时，cuda开始加速30%；32M时，cuda加速达到峰值100%。

CUDAPrimer 0.466, 2012-09-21 ----------------------- 
- 64M个整数，256M数据，gpu总耗时100ms。- 细分：内存拷贝至显存耗时95， kernel计算耗时5%，显存拷贝至内存以及内存后续计算几乎不耗时间。

CUDAPrimer 0.465, 2012-09-21 ----------------------- 
- 准备测量gpu耗时，先总后分，分为：内显拷贝、kernel、显内拷贝、后期cpu。

CUDAPrimer 0.464, 2012-09-21 ----------------------- 
- 基于cuda的gpu算法相对cpu算法，耗费的时间减少了吗？- kernel局部变量占用寄存器的规律，对性能有多大影响？

CUDAPrimer 0.463, 2012-09-17 ----------------------- 
- branch-chapter04-further
- 通过穷举实践，找到最优线程数目配置：块数32，每块线程数256，每线程分配元素个数128

CUDAPrimer 0.462, 2012-09-17 ----------------------- 
- branch-chapter04-further
- 分支上传服务器，再从别的地方下载，合作完善分支，准备最终合并到master主干

CUDAPrimer 0.461, 2012-09-17 ----------------------- 
- branch-chapter04-further
- 从第四章分出分支，做进一步的扩展实验。线程数目如何最优化配置？

CUDAPrimer 0.46, 2012-09-17 ----------------------- 
- 版本编号

CUDAPrimer 0.45, 2012-09-15 ----------------------- 
- Step05-UpgradeFurther chapter04-App1UpgradeOfCUDA
- 第四章第五步-进一步改善 - 效率会变差的一个原因是，在这一版的程序中，最后加总的工作，只由每个 block 的 thread 0 来进行，但这并不是最有效率的方法。理论上，把 256 个数字加总的动作，是可以并行化的。最常见的方法，是透过树状的加法。- 上一个版本的树状加法是一般的写法，但是它在 GPU 上执行的时候，会有 share memory 的 bank conflict 的问题（详情在后面介绍 GPU 架构时会提到）如果还要再提高效率，可以把树状加法整个展开。- 优化树状加法，将树倒置。耗时：0.1273M t，比上一环节0.128提升1.005倍。- 优化树状加法进一步，把树状加法整个展开。耗时：0.1262M t，比上一环节0.1273提升1.01倍。- 最终耗时，0.1262M t， 比Step04的0.1280上升1.014倍；带宽达到：50.51G。			比Step03的0.1226下降0.971倍；带宽达到：50.51G。- 发现峰值：Step03-ParallelFurther耗时0.1226Mt，带宽52G。

CUDAPrimer 0.44, 2012-09-14-----------------------
- Step04-syncthreads-tree chapter04-App1UpgradeOfCUDA
- 第四章第四步-Thread 的同步 在 CPU 上执行的部份，需要的时间加长了（因为 CPU 现在需要加总 8192 个数字）。为了避免这个问题，我们可以让每个block把自己的每个thread的计算结果进行加总。由于在 GPU 上多做了一些动作，所以它的效率会比较差一些。- 线程块内部，相加得到整块的和。耗时0.15M t，比上一环节降为0.8倍。- 线程块内部，采用树状加法。耗时0.128M t，比上一环节提升1.15倍。- 最终耗时，0.1280M t，比Step03的0.1226下降0.958倍；带宽降为0.958倍，达到：49.80G。

CUDAPrimer 0.43, 2012-09-14-----------------------
- Step03-ParallelFurther	chapter04-App1UpgradeOfCUDA 
- 第四章第三步-更多的并行化理论上 256 个 threads 最多只能隐藏 256 cycles 的 latency。但是 GPU 存取 global memory 时的 latency 可能高达 500 cycles 以上。如果增加 thread 数目，就可以看到更好的效率。- block 块并行，耗时： 0.1226M t， 比上一环节提升20倍带宽提升20倍，达到：52G

CUDAPrimer 0.42, 2012-09-14-----------------------
- Step02-Coalesce chapter04-App1UpgradeOfCUDA 
- 第四章第二步-内存的存取模式耗时： 2.53M t， 比上一环节提升4倍带宽提升4倍，达到： 2.6G

CUDAPrimer 0.41, 2012-09-14-----------------------
- Step01-Parallel chapter04-App1UpgradeOfCUDA 
- 第四章第一步-程序的并行化1M个数耗时10M t， 速度提升67*67倍；带宽原先是： 数据量 * 频率/耗时周期数  = 4M * 1.62/0.67 = 9.7M提速后带宽是：648M

CUDAPrimer 0.40, 2012-09-13-----------------------
- chapter04-App1UpgradeOfCUDA 
- 第四章沿用第三章源码 - 直接拷贝第三章源码

CUDAPrimer 0.33, 2012-09-11-----------------------
- chapter03-App1OfCUDA 
- 第三章 思维发散- 估算gflops 每秒浮点运算次数，选择gflops最大的cuda设备。- gts 250 ： 207.36 gflops

CUDAPrimer 0.32, 2012-09-11 -----------------------
- step03-timer chapter03-App1OfCUDA
- 第三章第三步，计算运行时间-	clock_t start = clock();    	...    	*time = clock() - start;

CUDAPrimer 0.31, 2012-09-11  -----------------------
- step02-sumOfSquares  chapter03-App1OfCUDA
- 第三章第二步，利用CUDA进行运算，求平方和

CUDAPrimer 0.30, 2012-09-11 -----------------------
- step01-InitializeCUDAProject   chapter03-App1OfCUDA- CUDA 的初始化
- 使用SDK提供cuda工程模板template_runtime，快速新建一个新工程sumOfSquares

CUDAPrimer 0.20, 2012-09-11 -----------------------
- chapter02-SetupCUDA
- 第二章总结：链接库值不值得用？比如：cutil。实验可以用，工程应用避免使用。

CUDAPrimer 0.11, 2012-09-11 -----------------------
- chapter01-WhatIsCUDA
- 第一章总结- 一个具有很少量执行单元的显示芯片，可能会把各个 block 中的 thread 顺序执行，而非同时执行。- 并行化执行的方式来隐藏内存的 latency的原理：当第一个 thread 需要等待内存读取结果时，则开始执行第二个 thread，依此类推。- 内存显存频繁交换数据影响效率的原因：由于 CPU 存取显卡内存时只能透过 PCI Express 接口，因此速度较慢（PCI Express x16 的理论带宽是双向各 4GB/s），因此不能太常进行这类动作，以免降低效率。

CUDAPrimer 0.10, 2012-09-11 -----------------------
- 选择教程：《深入浅出谈CUDA》 详见：http://www.cppblog.com/hktk/category/11855.html
- 版本编号规范参考：https://github.com/soulsheng/drupal/blob/7.x/CHANGELOG.txt