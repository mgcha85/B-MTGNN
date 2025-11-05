import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import random
from util import DataLoaderS
from trainer import Optim
from random import randrange
from matplotlib import pyplot as plt
import time
from safetensors.torch import save_file, load_file
from safetensors import safe_open
import pandas as pd
import json


def load_model(Data):
    with safe_open(args.save, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    arch_json = metadata.get("arch")
    if arch_json:
        arch = json.loads(arch_json)
        # 저장된 아키텍처로 모델 재생성
        model = gtnet(
            arch["gcn_true"],
            arch["buildA_true"],
            arch["gcn_depth"],
            arch["num_nodes"],
            device,
            Data.adj,
            dropout=arch["dropout"],
            subgraph_size=arch["subgraph_size"],
            node_dim=arch["node_dim"],
            dilation_exponential=arch["dilation_exponential"],
            conv_channels=arch["conv_channels"],
            residual_channels=arch["residual_channels"],
            skip_channels=arch["skip_channels"],
            end_channels=arch["end_channels"],
            seq_length=arch["seq_length"],
            in_dim=arch["in_dim"],
            out_dim=arch["out_dim"],
            layers=arch["layers"],
            propalpha=arch["propalpha"],
            tanhalpha=arch["tanhalpha"],
            layer_norm_affline=arch.get("layer_norm_affline", False),
        ).to(device)

        ret = model.load_state_dict(state_dict, strict=True)
        if hasattr(ret, "missing_keys") and ret.missing_keys:
            print("[load_state_dict] missing keys:", ret.missing_keys)
        if hasattr(ret, "unexpected_keys") and ret.unexpected_keys:
            print("[load_state_dict] unexpected keys:", ret.unexpected_keys)
    else:
        # 메타데이터가 없으면, 부분 로드(최소한 충돌은 방지)
        print("[warn] No arch metadata in checkpoint. Falling back to partial load (strict=False).")
        msd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in msd and v.shape == msd[k].shape}
        msd.update(filtered)
        model.load_state_dict(msd, strict=False)
        print(f"[warn] Loaded {len(filtered)} / {len(msd)} tensors by shape-matching.")
    return model

def inverse_diff_2d(output, I,shift):
    output[0,:]=torch.exp(output[0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[0]):
        output[i,:]= torch.exp(output[i,:]+torch.log(output[i-1,:]+shift))-shift
    return output

def inverse_diff_3d(output, I,shift):
    output[:,0,:]=torch.exp(output[:,0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[1]):
        output[:,i,:]=torch.exp(output[:,i,:]+torch.log(output[:,i-1,:]+shift))-shift
    return output


def plot_data(data,title):
    x=range(1,len(data)+1)
    plt.plot(x, data,'b-',label='Actual')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    plt.show()


# for figure display, we rename columns
def consistent_name(name):

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

#computes and saves validation/testing error to a text file given a single node's prediction and actual curve values
def save_metrics_1d(predict, test, title, type):
    #RRSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator

    #Relative Absolute Error RAE  - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))

    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s) # calculate the mean of each column in test
    diff_r = test_s - mean_all # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
    
    #Relative Absolute Error RAE
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()


    title=title.replace('/','_')
    with open('model/Bayesian/'+type+'/'+title+'_'+type+'.txt',"w") as f:
      f.write('rse:'+str(rrse)+'\n')
      f.write('rae:'+str(rae)+'\n')
      f.close()


def generate_month_labels(start: str, end: str):
    """
    Generate month-year labels between start and end inclusive.
    start, end format: 'Jul-11' or 'Jul-2011'
    """
    # pandas가 자동으로 날짜 파싱해줌
    start_date = pd.to_datetime(start, format='%b-%y', errors='coerce')
    if pd.isna(start_date):
        start_date = pd.to_datetime(start, format='%b-%Y')
    end_date = pd.to_datetime(end, format='%b-%y', errors='coerce')
    if pd.isna(end_date):
        end_date = pd.to_datetime(end, format='%b-%Y')

    # month 시작 기준으로 date_range 생성
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # 'MS' = Month Start
    # 'Mon-YY' 형식으로 변환
    return [d.strftime('%b-%y') for d in dates]

#plots predicted curve with actual curve. The x axis can be adjusted as needed
def plot_predicted_actual(predicted, actual, title, type, variance, confidence_95, M):
    # M = generate_month_labels("Jul-11", "Dec-24")
    M2=[]
    p=[]

    len_pred = len(predicted)
    #last 3 years
    if type=='Testing':
        M=M[-len_pred:]
        for index,value in enumerate(M):
            if 'Dec' in value or 'Mar' in value or 'Jun' in value or 'Sep' in value:
                M2.append(value)
                p.append(index+1) 
    
    else: ## last 3 years before Test data
        M=M[-2*len_pred:-len_pred]
        for index,value in enumerate(M):
            if 'Dec' in value or 'Mar' in value or 'Jun' in value or 'Sep' in value:
                M2.append(value)
                p.append(index+1) 

    pred_np = predicted.detach().cpu().numpy() if torch.is_tensor(predicted) else predicted
    act_np  = actual.detach().cpu().numpy() if torch.is_tensor(actual) else actual
    conf_np = confidence_95.detach().cpu().numpy() if torch.is_tensor(confidence_95) else confidence_95

    x = range(1, len(pred_np)+1)
    plt.plot(x, act_np, 'b-', label='Actual')
    plt.plot(x, pred_np, '--', color='purple', label='Predicted')
    plt.fill_between(x, pred_np - conf_np, pred_np + conf_np, alpha=0.5, color='pink', label='95% Confidence')
    
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(ticks = p ,labels = M2, rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    title=title.replace('/','_')
    plt.savefig('model/Bayesian/'+type+'/'+title+'_'+type+'.png', bbox_inches="tight")
    plt.savefig('model/Bayesian/'+type+'/'+title+'_'+type+".pdf", bbox_inches="tight", format='pdf')

    plt.show(block=False)
    plt.pause(2)
    plt.close()


#symmetric mean absolute percentage error (optional)
def s_mape(yTrue,yPred):
  mape=0
  for i in range(len(yTrue)):
    mape+= abs(yTrue[i]-yPred[i])/ (abs(yTrue[i])+abs(yPred[i]))
  mape/=len(yTrue)

  return mape

#for testing the model on unseen data, a sliding window can be used when the output period of the model is smaller than the target period to be forecasted.
#The sliding window uses the output from previous step as input of the next step.
#In our case, the window was not slided (we predicted 36 months and the model by default predicts 36 months)
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot, z=1.96):
    dev = current_device_of_model(model)
    test_window = test_window.to(dev)

    #model.eval()# To get Bayesian estimation, we must comment out this line
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    predictions = []
    sum_squared_diff=0
    sum_absolute_diff=0
    r=random.randint(0, 141)
    r=0 # we can choose any random node index for printing
    print('testing r=',str(r))
    scale = data.scale.expand(test_window.size(0), data.m) #scale will have the max of each column (142 max values)
    print('Test Window Feature:',test_window[:,r])
    
    x_input = test_window[0:n_input, :].clone() # Generate input sequence

    for i in range(n_input, test_window.shape[0], data.out_len):

        print('**************x_input*******************')
        print(x_input[:,r])#prints 1 random column in the sliding window
        print('**************-------*******************')

        X = torch.unsqueeze(x_input,dim=0)
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        X = X.to(torch.float).to(dev)


        y_true = test_window[i:i+data.out_len, :].clone().to(dev)  # ★ 타깃 정렬



        # Bayesian estimation
        num_runs = 10

        # Create a list to store the outputs
        outputs = []


        # Use model to predict next time step
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)  
                y_pred = output[-1, :, :,-1].clone()
                #if this is the last predicted window and it exceeds the test window range
                if y_pred.shape[0]>y_true.shape[0]:
                    y_pred=y_pred[:-(y_pred.shape[0]-y_true.shape[0]),]
            outputs.append(y_pred)

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)


        y_pred=torch.mean(outputs,dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        confidence = z * std_dev / torch.sqrt(
            torch.tensor(num_runs, device=std_dev.device, dtype=std_dev.dtype)  # ★ 상수 정렬
        )
        #shift the sliding window
        if data.P <= data.out_len:
            x_input = y_pred[-data.P:].clone().to(dev)
        else:
            x_input = torch.cat([x_input[-(data.P-data.out_len):, :].clone(),
                                y_pred.clone()], dim=0).to(dev)


        print('----------------------------Predicted months',str(i-n_input+1),'to',str(i-n_input+data.out_len),'--------------------------------------------------')
        print(y_pred.shape,y_true.shape)
        y_pred_o=y_pred
        y_true_o=y_true
        for z in range(y_true.shape[0]):
            print(y_pred_o[z,r],y_true_o[z,r]) #only one col
        print('------------------------------------------------------------------------------------------------------------')


        if predict is None:
            predict = y_pred
            test = y_true
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))
            variance=torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))


    dev_all = predict.device
    scale = data.scale.expand(test.size(0), data.m).to(dev_all)  # scale도 같은 dev

    predict = predict.to(dev_all) * scale
    test = test.to(dev_all) * scale
    variance = variance.to(dev_all) * scale
    confidence_95 = confidence_95.to(dev_all) * scale

    #Relative Squared Error RSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    #Relative Absolute Error RAE - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))# numerator


    #Root Relative Squared Error RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #Root Relative Squared Error RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=0) # calculate the mean of each column in test call it Yj-
    diff_r = test_s - mean_all.expand(test_s.size(0), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r
    print('rrse=',root_sum_squared,'/',root_sum_squared_r)

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    #Relative Absolute Error RAE
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()
###########################################################################################################


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g) #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for z in range(Ytest.shape[1]):
        smape+=s_mape(Ytest[:,z],predict[:,z])
    smape/=Ytest.shape[1]

    #plot predicted vs actual and save errors to file
    counter=0
    num_features = len(data.col)

    if is_plot:
        for v in range(r,r+num_features):
            col=v%data.m
            
            node_name=data.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name=consistent_name(node_name)
            
            #save error to file
            save_metrics_1d(torch.from_numpy(predict[:,col]),torch.from_numpy(Ytest[:,col]),node_name,'Testing')
            #plot
            plot_predicted_actual(predict[:,col], Ytest[:,col], node_name, 'Testing', variance[:,col], confidence_95[:,col], data.timeindex)
            counter+=1

    return rrse,rae,correlation, smape



def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot, z=1.96):
    #model.eval()# To get Bayesian estimation, we must comment out this line
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    sum_squared_diff=0
    sum_absolute_diff=0
    r=0 #we choose any node index for printing (debugging)
    # Bayesian estimation
    num_runs = 10
    print('validation r=',str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2,3)
        X = X.to(device)
        Y = Y.to(device)

        # Create a list to store the outputs
        outputs = []

        # Run the model multiple times (10)
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = torch.squeeze(output)
                if len(output.shape) == 1 or len(output.shape) == 2:
                    output = output.unsqueeze(dim=0)
                outputs.append(output)
            

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)

        # Calculate mean, variance, and standard deviation
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        confidence = z * std_dev / torch.sqrt(
            torch.tensor(num_runs, device=std_dev.device, dtype=std_dev.dtype)  # ★ 상수도 같은 device/dtype
        )

        output=mean #we will consider the mean to be the prediction

        scale = data.scale.expand(Y.size(0), Y.size(1), data.m).to(Y.device)

        #inverse normalisation
        output*=scale
        Y*=scale
        var*=scale
        confidence*=scale

        if predict is None:
            predict = output
            test = Y
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance= torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))


        print('EVALUATE RESULTS:')
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (142 max values)
        y_pred_o=output
        y_true_o=Y
        for z in range(Y.shape[1]):
            print(y_pred_o[0,z,r],y_true_o[0,z,r]) #only one col
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        #RRSE according to Lai et.al
        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        #Relative Absolute Error RAE - numerator
        sum_absolute_diff+=torch.sum(torch.abs(Y - output))

    #The below 2 lines are not used
    rse = math.sqrt(total_loss / n_samples) / data.rse 
    rae = (total_loss_l1 / n_samples) / data.rae 

    #RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=(0,1)) # calculate the mean of each column in test
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r #RRSE

    #Relative Absolute Error RAE
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    rae=sum_absolute_diff/sum_absolute_r # RAE
    rae=rae.item()


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0)/ (sigma_p * sigma_g) #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape+=s_mape(Ytest[x,:,z],predict[x,:,z])
    smape/=Ytest.shape[0]*Ytest.shape[2]


    #plot actual vs predicted curves and save errors to file
    counter=0
    num_features = len(data.col)
    if is_plot:
        for v in range(r,r+num_features):
            col=v%data.m
            node_name=data.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name=consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict[-1,:,col]),torch.from_numpy(Ytest[-1,:,col]),node_name,'Validation')
            plot_predicted_actual(predict[-1,:,col],Ytest[-1,:,col],node_name, 'Validation', variance[-1,:,col], confidence_95[-1,:,col], data.timeindex)
            counter+=1
    return rrse, rae, correlation, smape


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        X = X.to(device)
        Y = Y.to(device)

        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :] 
            ty = Y[:, :, :] 
            output = model(tx)           
            output = torch.squeeze(output,3)
            
            scale = data.scale.expand(output.size(0), output.size(1), data.m).to(output.device)
            scale = scale[:,:,:] 
            
            output*=scale 
            ty*=scale


            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            grad_norm = optim.step()

        if iter%1==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * output.size(1)* data.m)))
        iter += 1
    return total_loss / n_samples


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 어디서든 쓸 수 있게
def current_device_of_model(model):
    return next(model.parameters()).device

def resolve_device(dev_str: str) -> torch.device:
    dev_str = (dev_str or "").lower().strip()

    # 1) 자동 선택
    if dev_str in ("", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2) CPU 강제
    if dev_str == "cpu":
        return torch.device("cpu")

    # 3) CUDA 계열
    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

        # 'cuda'만 주면 0번으로
        if dev_str == "cuda":
            return torch.device("cuda:0")

        # 'cuda:<idx>' 파싱
        parts = dev_str.split(":")
        if len(parts) == 2:
            try:
                idx = int(parts[1])
            except ValueError:
                print(f"[warn] Invalid CUDA device spec '{dev_str}'. Falling back to cuda:0.")
                idx = 0
        else:
            idx = 0

        count = torch.cuda.device_count()
        if idx < 0 or idx >= count:
            if count > 0:
                print(f"[warn] Requested {dev_str} but only {count} CUDA device(s) available. Using cuda:0 instead.")
                return torch.device("cuda:0")
            else:
                print("[warn] No CUDA devices available. Falling back to CPU.")
                return torch.device("cpu")

        return torch.device(f"cuda:{idx}")

    # 4) 그 외 문자열은 CPU로
    print(f"[warn] Unknown device '{dev_str}'. Using CPU.")
    return torch.device("cpu")


def main(experiment):
    # Set fixed random seed for reproducibility
    set_random_seed(fixed_seed)

    #model hyper-parameters
    gcn_depths=[1,2,3]
    lrs=[0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]#[0.00001,0.0001,0.0002,0.0003]
    convs=[4,8,16]
    ress=[16,32,64]
    skips=[64,128,256]
    ends=[256,512,1024]
    layers=[1,2]
    ks=[20,30,40,50,60,70,80,90,100]
    dropouts=[0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs=[1,2,3]
    node_dims=[20,30,40,50,60,70,80,90,100]
    prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]


    best_val = 10000000
    best_rse=  10000000
    best_rae=  10000000
    best_corr= -10000000
    best_smape=10000000
    
    best_test_rse=10000000
    best_test_corr=-10000000

    best_hp=[]


    #random search
    for q in range(args.search_iters):

        #hps
        gcn_depth=gcn_depths[randrange(len(gcn_depths))]
        lr=lrs[randrange(len(lrs))]
        conv=convs[randrange(len(convs))]
        res=ress[randrange(len(ress))]
        skip=skips[randrange(len(skips))]
        end=ends[randrange(len(ends))]
        layer=layers[randrange(len(layers))]
        k=ks[randrange(len(ks))]
        dropout=dropouts[randrange(len(dropouts))]
        dilation_ex=dilation_exs[randrange(len(dilation_exs))]
        node_dim=node_dims[randrange(len(node_dims))]
        prop_alpha=prop_alphas[randrange(len(prop_alphas))]
        tanh_alpha=tanh_alphas[randrange(len(tanh_alphas))]
        

        Data = DataLoaderS(
            args.data, 
            args.train_ratio, 
            args.valid_ratio, 
            device, 
            args.horizon, 
            args.seq_in_len, 
            args.normalize, 
            args.seq_out_len
        )


        print('train X:', Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:', Data.valid[0].shape)
        print('valid Y:', Data.valid[1].shape)
        print('test X:', Data.test[0].shape)
        print('test Y:', Data.test[1].shape)
        print('test window:', Data.test_window.shape)

        print('length of training set=', Data.train[0].shape[0])
        print('length of validation set=', Data.valid[0].shape[0])
        print('length of testing set=', Data.test[0].shape[0])
        print('valid=', int((args.seq_out_len) * Data.n))

        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                    device, Data.adj, dropout=dropout, subgraph_size=k,
                    node_dim=node_dim, dilation_exponential=dilation_ex,
                    conv_channels=conv, residual_channels=res,
                    skip_channels=skip, end_channels= end,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False).to(device)
        

        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.L1Loss:
            criterion = nn.L1Loss(reduction='sum').to(device)
        else:
            criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
        evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE

        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
        )
        
        es_counter=0 #early stopping
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:',(experiment+1))
                print('Iter:',q)
                print('epoch:',epoch)
                print('hp=',[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=',best_val)
                print('best rrse=',best_rse)
                print('best rrae=',best_rae)
                print('best corr=',best_corr)
                print('best smape=',best_smape)       
                print('best hps=',best_hp)
                print('best test rse=',best_test_rse)
                print('best test corr=',best_test_corr)

                
                es_counter+=1 # feel free to use this for early stopping (not used)

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                 args.batch_size,False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape), flush=True)
                # Save the model if the validation loss is the best we've seen so far.
                sum_loss=val_loss+val_rae-val_corr
                if (not math.isnan(val_corr)) and val_loss < best_rse:
                    arch_meta = {
                        "gcn_true": args.gcn_true,
                        "buildA_true": args.buildA_true,
                        "gcn_depth": gcn_depth,
                        "num_nodes": args.num_nodes,
                        "dropout": dropout,
                        "subgraph_size": k,
                        "node_dim": node_dim,
                        "dilation_exponential": dilation_ex,
                        "conv_channels": conv,
                        "residual_channels": res,
                        "skip_channels": skip,
                        "end_channels": end,
                        "seq_length": args.seq_in_len,
                        "in_dim": args.in_dim,
                        "out_dim": args.seq_out_len,
                        "layers": layer,
                        "propalpha": prop_alpha,
                        "tanhalpha": tanh_alpha,
                        "layer_norm_affline": False  # net.py에서 그대로 사용
                    }
                    save_file(model.state_dict(), args.save, metadata={"arch": json.dumps(arch_meta)})

                    best_val = sum_loss
                    best_rse= val_loss
                    best_rae= val_rae
                    best_corr= val_corr
                    best_smape=val_smape

                    best_hp=[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]
                    
                    es_counter=0
                    
                    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                           args.seq_in_len, False) 
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse=test_acc
                    best_test_corr=test_corr

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=',best_val)
    print('best hps=',best_hp)
    
    #save best hp to desk
    with open(args.hp_path, "w") as f:
        f.write(str(best_hp))
    

    model = load_model(Data)
    
    # 4) 모델을 디바이스로 이동
    model = model.to(device)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, True)

    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                         args.seq_in_len, True) 
    print('********************************************************************************************************')    
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape


plt.rcParams['savefig.dpi'] = 1200

# parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
# parser.add_argument('--data', type=str, default='./data/Smoothed_CyberTrend_Forecasting_All.txt',
#                     help='location of the data file')
# parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
#                     help='report interval')
# parser.add_argument('--save', type=str, default='model/Bayesian/model.safetensors',
#                     help='path to save the final model')
# parser.add_argument('--optim', type=str, default='adam')
# parser.add_argument('--L1Loss', type=bool, default=True)
# parser.add_argument('--normalize', type=int, default=2)
# parser.add_argument('--device',type=str,default='cuda:0',help='')
# parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
# parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
# parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
# parser.add_argument('--num_nodes',type=int,default=142,help='number of nodes/variables')
# parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
# parser.add_argument('--subgraph_size',type=int,default=20,help='k')
# parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
# parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
# parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
# parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
# parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
# parser.add_argument('--end_channels',type=int,default=64,help='end channels')
# parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
# parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
# parser.add_argument('--seq_out_len',type=int,default=36,help='output sequence length')
# parser.add_argument('--horizon', type=int, default=1) 
# parser.add_argument('--layers',type=int,default=5,help='number of layers')

# parser.add_argument('--batch_size',type=int,default=8,help='batch size')
# parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
# parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

# parser.add_argument('--clip',type=int,default=10,help='clip')

# parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
# parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

# parser.add_argument('--epochs',type=int,default=200,help='')
# parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
# parser.add_argument('--step_size',type=int,default=100,help='step_size')

# parser.add_argument('--search_iters', type=int, default=60,
#                     help='number of random search iterations (default: 60)')
# parser.add_argument('--train_ratio', type=float, default=0.43,
#                     help='training split ratio (default: 0.43)')
# parser.add_argument('--valid_ratio', type=float, default=0.30,
#                     help='validation split ratio (default: 0.30)')
# parser.add_argument('--hp_path', type=str, default='model/Bayesian/hp.txt',
#                     help='path to save best hyperparameters (default: model/Bayesian/hp.txt)')

# args = parser.parse_args()


from config import get_args
args = get_args()

if args.train_ratio + args.valid_ratio > 1.0:
    raise ValueError(f"train_ratio + valid_ratio must be <= 1.0 "
                    f"(got {args.train_ratio + args.valid_ratio})")

fixed_seed = 123
device = resolve_device(args.device)
torch.set_num_threads(3)


if __name__ == "__main__":

    vacc = []
    vrae = []
    vcorr = []
    vsmape=[]
    acc = []
    rae = []
    corr = []
    smape=[]
    for i in range(1):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print('\n\n')
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))

