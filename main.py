import numpy as np
import imageio
import matplotlib.pyplot as plt
import scipy
import cv2
import time


MAX_STROKE=100
MIN_STROKE=10
f_c=0.9
def blur(img,R):
    res=cv2.blur(img,ksize=(R,R))
    return res

def makeStroke(canvas,R,x0,y0,ref,drefx,drefy):
    stroke_color=ref[x0,y0][:3]

    K={
        'strokes':[(x0,y0)],
        'color':stroke_color
    }
    if (stroke_color == np.array([0, 0, 0])).any():
        return K

    x,y=x0,y0
    lastDx,lastDy=0,0
    for point_id in range(MAX_STROKE):
        try:
            x,y=int(x),int(y)
        except:
            return K
        if x<0 or x>=ref.shape[0] or y<0 or y>=ref.shape[1]:
            return K
        # print(x,y)

        d_ref_canv=np.sum((ref[x,y,:3]-canvas[x,y,:3])**2)

        d_ref_stroke=np.sum((ref[x,y,:3]-stroke_color)**2)

        if point_id>MIN_STROKE and d_ref_canv<d_ref_stroke:
            return K

        if not drefx.any() and not drefy.any():
            return K

        gx,gy=drefx[x,y],drefy[x,y]

        # dx,dy=-gy,gx
        # print(f"dx:{dx},dy:{dy}")
        dx,dy=gy,gx
        if lastDx*dx+lastDy*dy<0:
            dx,dy=-dx,-dy
        dx,dy=f_c*dx+(1-f_c)*lastDx,f_c*dy+(1-f_c)*lastDy
        dx,dy=dx/np.sqrt(dx**2+dy**2),dy/np.sqrt(dx**2+dy**2)
        x,y=x+R*dx,y+R*dy
        lastDx,lastDy=dx,dy
        K['strokes'].append((x,y))

    return K



T=5000
def paintLayer(canvas,ref,R):
    S=[]
    D=np.sum((canvas[:,:,:3]-ref[:,:,:3])**2,axis=2)
    grid=R
    H,W,C=ref.shape

    gs_ref=np.sum(ref[:,:,:3]*np.array([0.299,0.587,0.114]),axis=2)
    REF_GRAD_X=cv2.Sobel(gs_ref,cv2.CV_64F,1,0,ksize=5)
    REF_GRAD_Y=cv2.Sobel(gs_ref, cv2.CV_64F, 0, 1, ksize=5)

    for i in range(grid//2,H-grid//2,grid):
        for j in range(grid//2, W-grid//2,grid):
            area_err=np.sum(D[i-grid//2:i+grid//2,j-grid//2:j+grid//2],axis=(0,1))//grid**2
            if area_err>T:
                xy=np.argmax(D[i-grid//2:i+grid//2,j-grid//2:j+grid//2])
                x1,y1=xy//grid-grid//2+i,xy%grid+j-grid//2
                S.append(makeStroke(canvas,R,x1,y1,ref,REF_GRAD_X,REF_GRAD_Y))


    for stroke in S:
        strokes,color=stroke['strokes'],stroke['color']
        if (color==np.array([0,0,0])).all():
            continue
        for x,y in strokes:
            try:
                x=int(x)
                y=int(y)
            except:
                continue
            if grid//2<=x<=canvas.shape[0]-grid//2 and grid//2<=y<canvas.shape[1]-grid//2:
                canvas[x-grid//2:x+grid//2,y-grid//2:y+grid//2]=color

    plt.imshow(canvas.astype(np.int32))
    plt.show()
    return canvas


def paint(sourceImg,Rs):
    H,W,C=sourceImg.shape
    canvas=np.zeros((H,W,3))
    for R in sorted(Rs,reverse=True):
        ref_img=blur(sourceImg,R)
        canvas=paintLayer(canvas,ref_img,R)
    return canvas


if __name__ == '__main__':

    img=imageio.imread('ironman.jpg')
    H,W,C=img.shape

    Rs=[int(r*min(H,W)) for r in [1e-2,2e-2,3.5e-2,5e-2,7.5e-2]]

    canvas=paint(img,Rs)

    plt.imshow(canvas)
    plt.show()