import mediapipe as mp
import cv2
import numpy as np
import time

#contants
ml = 150
max_x, max_y = 250+ml, 50
curr_tool = "select tool"
curr_color = (0,0,0)
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0
COLORX, COLORY = 500,100
THICKX, THICKY = 500,320

#get tools function
def getTool(x):
    if x < 50 + ml:
        return "line"

    elif x<100 + ml:
        return "rectangle"

    elif x < 150 + ml:
        return"draw"

    elif x<200 + ml:
        return "circle"

    else:
        return "erase"

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True

    return False

def getColor(y):
    if y < COLORY+30:
        return (0,0,255)

    elif y < COLORY+70:
        return (255,0,0)

    elif y < COLORY+110:
        return (0,255,0)

    elif y < COLORY+150:
        return (0,255,255)

    else:
        return (0,0,0)

def getThickness(x):
    if x < THICKX+30:
        return 2

    elif x < THICKX+60:
        return 4

    elif x < THICKX+90:
        return 6
    
    else:
        return 8

hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

maskRGB = np.ones((480, 640))*255
maskRGB = maskRGB.astype('uint8')

maskR = np.ones((480, 640))*255
maskR = maskR.astype('uint8')

maskG = np.ones((480, 640))*255
maskG = maskG.astype('uint8')

maskB = np.ones((480, 640))*255
maskB = maskB.astype('uint8')

def handleDrawing(mask,frm):
    op=cv2.bitwise_and(frm,frm,mask=mask)
    print(op[0,:,0])
    if curr_color[0] == 0:
        frm[:, :, 0] = np.minimum(frm[:, :, 0], op[:, :, 0])
    if curr_color[1] == 0:
        frm[:, :, 1] = np.minimum(frm[:, :, 1], op[:, :, 1])
    if curr_color[2] == 0:
        frm[:, :, 2] = np.minimum(frm[:, :, 2], op[:, :, 2])

cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    mask = np.ones((480, 640))*255
    mask = mask.astype('uint8')

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)
            # print(curr_color)

            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0,255,255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("your current tool set to : ", curr_tool)
                    time_init = True
                    rad = 40
            
            elif x < COLORX+120 and x > COLORX and y > COLORY-10 and y < COLORY+190:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0,255,255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_color = getColor(y)
                    print("your color set to : ", curr_color)
                    time_init = True
                    rad = 40

            elif x < THICKX+120 and x > THICKX and y > THICKY-10 and y < THICKY+10:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0,255,255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    thick = getThickness(x)
                    print("your thickness set to : ", thick)
                    time_init = True
                    rad = 40

            else:
                time_init = True
                rad = 40

            if curr_tool == "draw":
                xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
                y9  = int(i.landmark[9].y*480)

                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    prevx, prevy = x, y

                else:
                    prevx = x
                    prevy = y
                    

            elif curr_tool == "line":
                xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
                y9  = int(i.landmark[9].y*480)

                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)

                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
                y9  = int(i.landmark[9].y*480)

                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), curr_color, thick)
                        # handleDrawing(mask,frm)
                        var_inits = False

            elif curr_tool == "circle":
                xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
                y9  = int(i.landmark[9].y*480)

                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True
                        # print(mask)

                    cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
                        var_inits = False

            elif curr_tool == "erase":
                xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
                y9  = int(i.landmark[9].y*480)

                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0,0,0), -1)
                    cv2.circle(maskRGB, (x, y), 30, 255, -1)
                    cv2.circle(maskR, (x, y), 30, 255, -1)
                    cv2.circle(maskG, (x, y), 30, 255, -1)
                    cv2.circle(maskB, (x, y), 30, 255, -1)

    maskRGB = np.minimum(maskRGB,mask)
    if curr_color[0] == 0:
        maskB = np.minimum(maskB,mask)
    if curr_color[1] == 0:
        maskG = np.minimum(maskG,mask)
    if curr_color[2] == 0:
        maskR = np.minimum(maskR,mask)

    op = cv2.bitwise_and(frm, frm, mask=maskRGB)
    frm[:, :, 0] = np.minimum(maskB,frm[:,:,0])
    frm[:, :, 1] = np.minimum(maskG,frm[:,:,1])
    frm[:, :, 2] = np.minimum(maskR,frm[:,:,2])
    
    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

    cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(frm, "RED", (COLORX,COLORY+00), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frm, "BLUE", (COLORX,COLORY+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frm, "GREEN", (COLORX,COLORY+80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frm, "YELLOW", (COLORX,COLORY+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frm, "BLACK", (COLORX,COLORY+160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.putText(frm, "-", (THICKX+00,THICKY), cv2.FONT_HERSHEY_SIMPLEX, 1, curr_color, 2)
    cv2.putText(frm, "-", (THICKX+30,THICKY), cv2.FONT_HERSHEY_SIMPLEX, 1, curr_color, 4)
    cv2.putText(frm, "-", (THICKX+60,THICKY), cv2.FONT_HERSHEY_SIMPLEX, 1, curr_color, 6)
    cv2.putText(frm, "-", (THICKX+90,THICKY), cv2.FONT_HERSHEY_SIMPLEX, 1, curr_color, 8)
        
        
    cv2.imshow("paint app", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
 