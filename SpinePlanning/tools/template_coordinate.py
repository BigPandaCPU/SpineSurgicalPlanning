class Coordinate_Template:
    def init_c1(self):
        coord_origin = [-2.1985015325150314, -222.54419772182055, -768.3221528613522]

        spine_coordinate = [[ 0.995543226797888, 0.09417094761653517, -0.005051356436917149],
                            [-0.08231042744431405, 0.8938045360117294, 0.4408383433173566],
                            [0.04602908983222723, -0.43845784749489664, 0.8975723028588609]]

        pedicle_center_R = [-20.503053743384807, -222.17990257531008, -767.1616310798598]
        pedicle_center_L = [16.63730287398107, -219.8306447711571, -767.2171960006659]

        pedicle_normal_R = [0.3330280120186491, -0.8142408175925819, -0.475504189442186]
        pedicle_normal_L = [-0.18303804126233253, -0.8505587841433759, -0.49299779935349664]

        pedicle_lower_reference_point0 = [13.46057, -228.86993, -773.88861]
        pedicle_lower_reference_point1 = [-18.440462, -228.53723, -773.78223]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_c2(self):
        coord_origin = [ -2.754043026460927, -190.45930692858252, -367.66945101018666]

        spine_coordinate = [[0.9960592845814706, 0.07987851864748348, -0.038539899555299725],
                            [-0.05590968923388905, 0.902854939803254,0.426294574587377],
                            [0.06884771781852222, -0.42245991517721876, 0.903763028575322]]

        pedicle_center_R = [-18.04008007394066, -193.0199952358089, -364.9757138838417]
        pedicle_center_L = [13.129597471555801, -191.30722526220038, -364.7250997829854]

        pedicle_normal_R = [0.3330280120186491, -0.8142408175925819, -0.475504189442186]
        pedicle_normal_L = [-0.18303804126233253, -0.8505587841433759, -0.49299779935349664]

        pedicle_lower_reference_point0 = [9.69871581, -195.42300831, -371.03433622]
        pedicle_lower_reference_point1 = [-16.07401019, -196.24342831, -369.49445622]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_c3(self):
        coord_origin = [-3.0810849183896525, -188.09772034998932, -384.82407486388837]

        spine_coordinate = [[0.9971689010194799, 0.07011920138999045, 0.027156591023777998],
                            [-0.07436127223187605, 0.865956596572124, 0.49455998023018455],
                            [0.011161721715738135, -0.49517923063235425, 0.8687191407576388]]

        pedicle_center_R = [-15.478027636347216, -192.2990364267877, -382.07407907874665]
        pedicle_center_L = [10.711353486241896, -190.9856590464166, -381.63076548011384]

        pedicle_normal_R = [ 0.3330280120186491, -0.8142408175925819, -0.475504189442186]
        pedicle_normal_L = [-0.18303804126233253, -0.8505587841433759, -0.49299779935349664]

        pedicle_lower_reference_point0 = [11.29583199, -191.84334583, -387.67058084]
        pedicle_lower_reference_point1 = [-15.57079201, -192.90924583, -387.43956084]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_c4(self):
        coord_origin = [-2.9172686723020993, -184.6289284250761, -401.05109058593143]

        spine_coordinate = [[0.9969630578703411, 0.07016092369942908, 0.033795059216722864],
                            [-0.07764052201220976, 0.861763686416715, 0.5013335198399816],
                            [0.0060506680201849175, -0.5024348649915854, 0.8645938907125107]]

        pedicle_center_R = [-15.533836244925906, -189.8231402736259, -399.41767478253485]
        pedicle_center_L = [10.655544877663205, -188.50976289325482, -398.97436118390203]

        pedicle_normal_R = [0.3330280120186491, -0.8142408175925819, -0.475504189442186]
        pedicle_normal_L = [-0.18303804126233253, -0.8505587841433759, -0.49299779935349664]

        pedicle_lower_reference_point0 = [10.92605452, -186.77260369, -403.60933352]
        pedicle_lower_reference_point1 = [-15.49405495, -187.45228322, -404.17028599]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_c5(self):
        coord_origin = [-2.5142520713243752, -182.2536296992382, -417.70294443324025]

        spine_coordinate = [[0.9972712672155106, 0.06447750115642518, 0.03595374015304786],
                            [-0.07382267964755812, 0.8741646470677245, 0.479985814151434],
                            [-0.00048120268463695414, -0.48133026256571565, 0.8765391872485767]]

        pedicle_center_R = [-19.086959997127934, -185.82159767651598, -415.7722737021147]
        pedicle_center_L = [13.318057855655267, - 179.048778157473, - 411.8954426474272]

        pedicle_normal_R = [0.32942002992634, - 0.8276902037516807, - 0.454325181446979]

        pedicle_normal_L = [- 0.1868055642514585, - 0.8610662143114862, - 0.47293620683553106]
        pedicle_lower_reference_point0 = [11.85098268, - 184.12610465, - 420.43536581]
        pedicle_lower_reference_point1 = [-15.18098132, - 184.72554465, - 419.61523581]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_c6(self):
        coord_origin = [-2.818518462292024, - 180.11351462693057, - 434.5615858764592]

        spine_coordinate = [[0.9981159595209652, 0.04585627308906254, 0.04076436639915133],
                            [-0.05750447855361273, 0.9308319079929098, 0.36089499027910604],
                            [-0.02139547372273081, -0.3625591831420879, 0.9317151240714739]]

        pedicle_center_R = [-18.413224917306902, - 183.5089931517395, - 432.0681825515897]
        pedicle_center_L = [ 12.86695243750572, - 180.71459007556763, - 430.06414660188267]

        pedicle_normal_R = [0.3138764805070229, - 0.8872461030514099, - 0.33804719730329325]
        pedicle_normal_L = [-0.2027863585825821, - 0.9109830566771532, - 0.35914838607456845]

        pedicle_lower_reference_point0 = [12.77404257, - 184.80741664, - 435.81726346]
        pedicle_lower_reference_point1 = [-17.25192359, - 185.19505154, - 436.63741599]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_c7(self):
        coord_origin = [-3.0115506319779515, - 178.0147777756071, - 451.6186739879511]

        spine_coordinate = [[0.9977413553348039, 0.04882568193093156, 0.046132858557085106],
                            [-0.06330022264863094, 0.9132331693968221, 0.40249007456839087],
                            [-0.022478204272188898, -0.4045012127267534, 0.9142611767078835]]

        pedicle_center_R = [-17.195986004114808, - 182.1069198926283, - 449.69377530291746]
        pedicle_center_L = [10.03437498068744, - 181.4411712964369, - 448.67074979022215]

        pedicle_normal_R = [0.3193777847132096, - 0.8694784873703671, - 0.3768354954510273]
        pedicle_normal_L = [-0.19709114498088792, - 0.8947525201180533, - 0.4007156202502162]

        pedicle_lower_reference_point0 = [10.27356393, - 181.81314008, - 453.46020876]
        pedicle_lower_reference_point1 = [-15.49898312, - 182.24475576, - 453.78158162]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_l1(self):
        coord_origin = [387.11847993702565, 283.472665065932, - 243.08765083933338]

        spine_coordinate = [[0.9865776659511668, 0.10406448712626687, 0.12583756023340117],
                            [-0.060789481065127764, 0.9492964232443379, -0.30844924997045986],
                            [-0.15155575884201927, 0.2966595411151541, 0.9428806757099044]]

        pedicle_center_R = [372.85361023321116, 285.6417924403846, - 241.349349885152466]
        pedicle_center_L = [400.1920489030516, 288.8987898667355, - 237.96590096649197]

        pedicle_normal_R = [0.31406321914847196, - 0.8900160608284403, 0.33050825382359966]
        pedicle_normal_L = [-0.19662695969343746, - 0.9438838032026482, 0.26536993946832]

        pedicle_lower_reference_point0 = [401.02967, 284.35033, - 244.32398]
        pedicle_lower_reference_point1 = [374.74015, 282.26503, - 247.61600]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_l2(self):
        coord_origin = [393.2297070373478, 275.567287607981, - 275.34871468731285]

        spine_coordinate = [[0.9745489024015432, 0.16859528242634403, 0.14775001716253283],
                            [-0.128872866701526, 0.9606319855602734, -0.24612592822914325],
                            [-0.1834290627356471, 0.22082078494133914, 0.9579102044982154]]

        pedicle_center_R = [378.6950329101288, 275.8270758072761, - 273.9862639415895]
        pedicle_center_L = [406.9851080388575, 281.17837890786205, - 269.8078859231236]

        pedicle_normal_R = [0.3767134465801897, - 0.8842635744056396, 0.27597990895178703]
        pedicle_normal_L = [ -0.1277501860703646, -0.9715349144183913, 0.19949887224001128]

        pedicle_lower_reference_point0 = [407.98564, 278.25476, - 275.631733]
        pedicle_lower_reference_point1 = [378.96866, 273.50580, - 279.50981]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_l3(self):
        coord_origin = [397.1328908318286, 267.3962026874078, - 309.2333784998693]

        spine_coordinate = [[0.9890456983936871, 0.13890561428773748, -0.049938330051076306],
                            [-0.14753145973786702, 0.941241811804015, -0.3038063858767776],
                            [0.004803631600990791, 0.3078458738252661, 0.9514241131546964]]

        pedicle_center_R = [380.2677574223982, 266.0797810319942, - 303.47776733027854]
        pedicle_center_L = [412.0280616826521, 270.1654347185176, - 304.8758081017629]

        pedicle_normal_R = [0.39848831037194143, - 0.8732183563552817, 0.28052944341208785]
        pedicle_normal_L = [-0.11347941607007819, - 0.9451211932539442, 0.3063794252077561]

        pedicle_lower_reference_point0 = [412.23328, 268.39715, - 309.78744]
        pedicle_lower_reference_point1 = [380.20547, 265.14932, - 309.18184]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_l4(self):
        coord_origin = [0.4937079413101627, 180.35479994640994, - 302.07649163591367]

        spine_coordinate = [[0.9805039074726379, 0.18252783717096377, -0.07277139608787765],
                            [-0.18765863628612156, 0.9796439336846451, -0.07128814362927322],
                            [0.05827798763588211, 0.08355448465449192, 0.9947976297977565]]

        pedicle_center_R = [-15.897682283427384, 178.4117154580559, - 295.63987076339737]
        pedicle_center_L = [16.12253159302438, 184.52920159435612, - 297.88157216197436]

        pedicle_normal_R = [0.4350374083663102, - 0.8990216955921968, 0.0500244357934777]
        pedicle_normal_L = [-0.07250876173640677, - 0.9935050566346313, 0.0876936822859613]

        pedicle_lower_reference_point0 = [17.96639, 183.12043, - 304.21891]
        pedicle_lower_reference_point1 = [-18.31880, 175.02475, - 301.27432]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_l5(self):
        coord_origin = [-2.176830135891319, 177.86552927944263, - 335.1787129746212]

        spine_coordinate = [[0.98023346, 0.17506565, -0.09216494],
                            [-0.17471933, 0.98454665, 0.01187615],
                            [0.09281978894703014, 0.004461594349386608, 0.9956729287048484]]

        pedicle_center_R = [-20.566719333131527, 174.20752542064358, -328.11050222283626]
        pedicle_center_L = [17.329173326691574, 180.40405868734686, -331.47358345554295]

        pedicle_normal_R = [0.45367745515542435, -0.890342404818564, -0.03830364037000037]
        pedicle_normal_L = [-0.05026326604074649, -0.9986939887829298, 0.009160832700064419]

        pedicle_lower_reference_point0 = [20.57510, 181.98001, - 338.44713]
        pedicle_lower_reference_point1 = [-22.19078, 172.89517, - 335.55464]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t1(self):
        coord_origin = [-3.025783817344241, - 171.7988700252188, - 468.7630190181614]

        spine_coordinate = [[0.9966633840790067, 0.061137198816869895, 0.05407718333094824],
                            [-0.07673778236845451, 0.927602681593271, 0.36560166555712614],
                            [-0.027810278555853725, -0.36853155634463586, 0.9291991607749366]]

        pedicle_center_R = [-17.403459343405018, - 177.76699304197095, - 466.3165870635625]
        pedicle_center_L = [10.45684674075102, - 176.57299470517896, - 465.12728271175587]

        pedicle_normal_R = [0.3320784711978154, - 0.8801719152679104, - 0.33914788594437667]
        pedicle_normal_L = [-0.18383245951413524, -0.911818858103961, - 0.36714029584747665]

        pedicle_lower_reference_point0 = [10.23769964, - 174.65748846, - 470.27756209]
        pedicle_lower_reference_point1 = [-17.17656636, - 176.22272846, - 471.80060209]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t2(self):
        coord_origin = [-3.176217558433315, - 162.4716764520213, - 486.35568203516937]

        spine_coordinate = [[0.9998566830469205, 0.015673719495566188, 0.006399053335020069],
                            [-0.016928847825188057, 0.9219362482769625, 0.38697153412662116],
                            [0.00016576405445695164, -0.38702440314555536, 0.9220694571951209]]

        pedicle_center_R = [-16.13475017442677, - 167.72028653020232, - 482.7410841195977]
        pedicle_center_L = [8.862276030636098, - 166.93369213933317, - 482.4281606882012]

        pedicle_normal_R = [0.2751339632692444, - 0.8864653752897214, - 0.372129601977875]
        pedicle_normal_L = [-0.24242994062191106, - 0.8945786895158159, - 0.37544199572533493]

        pedicle_lower_reference_point0 = [10.08367743, - 166.34239095, - 488.97019987]
        pedicle_lower_reference_point1 = [-15.97714, - 167.36056282, - 488.77940411]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t3(self):
        coord_origin = [-3.2133869676204383, - 153.19213318090087, - 505.0807991462601]

        spine_coordinate = [[0.9949158333465784, 0.10026045066060792, -0.009504030177524407],
                            [-0.08642848536799685, 0.8984617677347573, 0.43046087956506457],
                            [0.0516972095308581, -0.42745092578244226, 0.9025591972687751]]

        pedicle_center_R = [-14.39953106620396, - 157.83369235465972, - 501.00173220506554]
        pedicle_center_L = [7.954820176179361, - 156.01249738802562, - 501.2576526152218]

        pedicle_normal_R = [0.3409866720881351, - 0.8418981112868055, - 0.41825310479417654]
        pedicle_normal_L = [- 0.17401965980014508, - 0.8937967394898594, - 0.41333345676383176]

        pedicle_lower_reference_point0 = [9.561978, - 156.10192, - 507.77845]
        pedicle_lower_reference_point1 = [-14.96651, - 157.56573, - 506.88395]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t4(self):
        coord_origin = [-3.751806359715124, - 145.03856102843807, - 525.2417976103175]

        spine_coordinate = [[0.9996188654715517, 0.005103529474614064, -0.027130753404019662],
                            [0.004877689790307254, 0.934651373881803, 0.3555320202782979],
                            [0.02717226408817475, -0.3555288501482829, 0.9342702525375419]]

        pedicle_center_R = [-14.213375199042915, - 149.1102385457305, - 518.6738284330952]
        pedicle_center_L = [7.024602246082666, - 148.56089925129692, - 518.6562655668843]

        pedicle_normal_R = [0.25400891368672784, - 0.9014830099837212, - 0.350439516148477]
        pedicle_normal_L = [-0.2634318867688964, - 0.9041247912342656, - 0.3363956047705964]

        pedicle_lower_reference_point0 = [7.8857615, - 147.26637, - 527.47381]
        pedicle_lower_reference_point1 = [-15.19481, - 148.53761, - 527.25444]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t5(self):
        coord_origin = [-4.5183214956988325, - 138.33646004016583, - 547.0526904883751]

        spine_coordinate = [[0.9936237897144872, 0.09853548300676111, 0.05479528357482217],
                            [-0.11249408016175047, 0.8989403551123705, 0.4233808213405596],
                            [-0.007539657948642404, -0.42684540121579084, 0.9042931808981797]]

        pedicle_center_R = [-15.390724294178602, - 140.41767823572255, - 542.1148295347081]
        pedicle_center_L = [6.52338055492151, - 139.3208097207079, - 541.4106440488682]

        pedicle_normal_R = [0.365829697777919, - 0.8428068456759795, - 0.3947724067173672]
        pedicle_normal_L = [-0.1485078231121839, - 0.8938125649170305, - 0.42313653265928175]

        pedicle_lower_reference_point0 = [8.61556, - 138.85901, - 547.81366]
        pedicle_lower_reference_point1 = [-16.09912, - 140.02772, - 548.91736]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t6(self):
        coord_origin = [-16.457573369551255, - 139.41694177625158, - 392.59340274549425]

        spine_coordinate = [[0.9961087840609691, 0.05231615584121391, -0.07092467944640743],
                            [-0.021598312606230685, 0.9251211410178578, 0.3790572349057338],
                            [0.08544473774985419, -0.37605038795328327, 0.9226512355766093]]

        pedicle_center_R = [-26.138633151169238, - 143.21336985511255, - 384.0138189436735]
        pedicle_center_L = [-4.540308733101304, - 142.07910395545434, - 385.4689123274626]

        pedicle_normal_R = [0.278674292259516, - 0.8800579850569024, - 0.3844978306456919]
        pedicle_normal_L = [-0.23694955635827009, - 0.9071388200534176, - 0.3477845150286488]

        pedicle_lower_reference_point0 = [-4.620287, - 140.47172, - 392.43304]
        pedicle_lower_reference_point1 = [-25.80519, - 141.41446, - 391.16048]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx] + pedicle_lower_reference_point1[idx]) / 2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t7(self):
        coord_origin = [-17.272177963809728, - 132.20707367843724, - 414.8500628137687]

        spine_coordinate = [[0.9990145895168093, -0.0214466821131289, -0.03885729994350388],
                            [0.029495391421858957, 0.9750137075896602, 0.22017786423057178],
                            [0.033164315422338476, -0.22110700992642268, 0.9746854971445735]]

        pedicle_center_R = [-27.750938992947482, - 131.17169838601964, - 407.41289711845525]
        pedicle_center_L = [-6.718406022836905, - 131.37145373676836, - 408.3180637444318]

        pedicle_normal_R = [0.23007364177134867, - 0.9473417309318479, - 0.22273249470411727]
        pedicle_normal_L = [-0.2870543624331059, - 0.9362401113615733, - 0.20261847617083745]

        pedicle_lower_reference_point0 = [-6.184613, - 133.12459, - 414.23195]
        pedicle_lower_reference_point1 = [-28.29848, - 132.25341, - 412.83391]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx] + pedicle_lower_reference_point1[idx]) / 2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R, \
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t8(self):
        coord_origin = [378.1579690824297, 324.4409441029702, - 105.25030947552223]

        spine_coordinate = [[0.9952254962548119, 0.0895339886589819, 0.038857129065031575],
                            [-0.08498019257334835, 0.9907049046566255, -0.10621750684093134],
                            [-0.048006025398483566, 0.10240828464592096, 0.9935833959770666]]

        pedicle_center_R = [366.1374129625759, 326.9842420491827, - 99.62371534958719]
        pedicle_center_L = [389.0381218423192, 329.26663645836237, - 98.54880777425235]

        pedicle_normal_R = [0.3396678753319684, - 0.933774352190146, 0.11265519810172825]
        pedicle_normal_L = [-0.1754987498727371, - 0.9801205550880212, 0.0925412680216545]

        pedicle_lower_reference_point0 = [390.50680, 327.83830, - 103.92784]
        pedicle_lower_reference_point1 = [366.54799, 324.15440, - 105.32784]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t9(self):
        coord_origin = [380.87624535259897, 319.6572245778264, - 129.6881003217194]

        spine_coordinate = [[0.9893167602496834, 0.13312340737972025, 0.05941806372378243],
                            [-0.10844713235335025, 0.9444323566297574, -0.31030105258457935],
                            [-0.09742467538260186, 0.30054231342420973, 0.9487796111153626]]

        pedicle_center_R = [367.89537876112314, 323.2861033193117, - 126.42759072533542]
        pedicle_center_L = [392.5606026245129, 326.7378974138212, - 124.88989982093004]

        pedicle_normal_R = [0.36080590511883226, - 0.8777967312729172, 0.3151063271309575]
        pedicle_normal_L = [-0.15130213326465272, - 0.9467064776305433, 0.28434927410129734]

        pedicle_lower_reference_point0 = [395.16965, 324.34486, - 129.81058]
        pedicle_lower_reference_point1 = [367.65252, 319.94450, - 131.01037]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t10(self):
        coord_origin = [382.302794365633, 312.70476121172226, - 154.84914240728838]

        spine_coordinate = [[0.995544265893549, 0.08101284120780616, -0.04825488789659934],
                            [-0.09232858283564596, 0.9414446709063952, -0.32427976257162705],
                            [0.01915848214331865, 0.3272901635878908, 0.9447296445971068]]

        pedicle_center_R = [369.5425245099756, 316.48075825380585, - 150.6159189379654]
        pedicle_center_L = [395.92351968819827, 319.0940776507785, - 151.9658311999283]

        pedicle_normal_R = [0.34684837892147835, - 0.888398055448253, 0.3007409136038947]
        pedicle_normal_L = [-0.16848325359023836, - 0.9303333878531462, 0.3257194816177488]

        pedicle_lower_reference_point0 = [397.87819, 316.18036, - 158.436201]
        pedicle_lower_reference_point1 = [368.80208, 314.04279, - 155.57476]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t11(self):
        coord_origin = [383.089417480883, 302.4608462913364, - 181.6098415783268]

        spine_coordinate = [[0.9946980014630905, 0.10253412172306205, 0.007914528906651971],
                            [-0.0933366915931105, 0.932420280525171, -0.3491141395988052],
                            [-0.043175778948230964, 0.3465244209977729, 0.9370467852590775]]

        pedicle_center_R = [369.36646572319074, 307.82627841556325, - 179.50064893741643]
        pedicle_center_L = [396.83624696301825, 310.70743406538423, - 179.23054993330672]

        pedicle_normal_R = [0.34760310785422605, - 0.8741110464401721, 0.33926679457522907]
        pedicle_normal_L = [-0.16729046595389968, - 0.9271866133897492, 0.3351699329471171]

        pedicle_lower_reference_point0 = [397.09312, 306.08623, - 186.12727]
        pedicle_lower_reference_point1 = [368.31043, 304.26839, - 185.91120]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid

    def init_t12(self):
        coord_origin = [383.4378415723599, 292.41128018247406, - 210.38854321127278]

        spine_coordinate = [[0.9993475549581664, 0.035839327453534214, 0.004472919271896806],
                            [-0.03218859908748996, 0.9399557963050063, -0.3397749182641453],
                            [-0.016381650952190296, 0.3394092567981906, 0.9404961445491313]]

        pedicle_center_R = [368.5733132565266, 296.76493021810484, - 207.4767290396371]
        pedicle_center_L = [398.1905706926432, 298.7119199133359, - 207.5278281810759]

        pedicle_normal_R = [0.28974197907048277, - 0.8986516787124721, 0.3293550453713682]
        pedicle_normal_L = [-0.2275583807291402, - 0.9172034797297526, 0.3270396919818222]

        pedicle_lower_reference_point0 = [398.39678, 292.54808, - 215.08185]
        pedicle_lower_reference_point1 = [370.36530, 290.57031, - 215.58760]
        pedicle_lower_reference_point_mid = []
        for idx in range(3):
            pedicle_lower_reference_point_mid.append(
                (pedicle_lower_reference_point0[idx]+pedicle_lower_reference_point1[idx])/2)
        return coord_origin, spine_coordinate, pedicle_center_L, pedicle_center_R,\
            pedicle_normal_L, pedicle_normal_R, pedicle_lower_reference_point_mid