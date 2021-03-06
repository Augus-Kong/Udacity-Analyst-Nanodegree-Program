# 泰坦尼克号数据可视化
##概要
泰坦尼克号是一艘奥林匹克级邮轮，于1912年4月处女航时撞上冰山后沉没，泰坦尼克号海难为和平时期死伤人数最惨重的海难之一。这个数据可视化将利用数据与你一起探索船上人员的生死与他们的性别、船舱级别之间的联系。
##设计
#### 初版思路
主要从两个因素切入，一个是船舱级别，还有一个是性别，因为船舱级别和性别对于人群的分割非常明显。对比各个船舱级别之间的的存活率和死亡率，对比同船舱级别内男女的存活率和死亡率，对比不同船舱级别之间男女的存活率和死亡率。用气泡表示，气泡的Y轴坐标表示存活率或死亡率，气泡的X轴坐标表示船舱级别或者性别。气泡与气泡使用直线连接，便于观察比较。设置2个选择按钮，一个选择为不区分男女乘客，一个选择为区分男女乘客。

####获取反馈后的改进
1. 
* 用红色代表死亡，绿色代表存活，符合生活中的习惯，区分显著。
* 显示存活人数和死亡人数的具体数值，并且让气泡的大小对应表示人数的多少。
* 为按钮添加了背景颜色，并且点击后会闪现。
2. 
*  改用柱状图，柱状图Y轴为百分比，人数则使用柱状图中数字来表示。
*  男女用紧邻的不同柱状图表示，同一船舱级别的男女在X轴上处于同一船舱级别范围中。


##反馈
1. 
* 代表存活和死亡的颜色没有足够的显著性和区分度。
* 没有给出存活人数和死亡人数的具体数值。
* 按钮的可辨识度较低，容易让人误以为是普通文本内容。
* 代表存活率和死亡率的气泡太小并且大小相同没有区别。
2. 
* 使用气泡+折线图的表示方式并不合适。
* 横轴间并没有明显的逻辑关系的表示，使用折现并不合理，对于该可视化建议使用柱状图。
*  男女分别用不同的柱状图表示。


##资源
* https://classroom.udacity.com/nanodegrees/nd002/parts/00213454010/modules/318423863275460/lessons/3068848585/concepts/30952087200923
*  https://discussions.youdaxue.com/c/dand-p6
*  http://dimplejs.org/
*  http://www.w3school.com.cn/
