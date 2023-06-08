%generative RVoxM with latent variables, case with no prior (deterministic
%model weights)

clear all, close all

n_set_tot=1;



for n_set=1:n_set_tot



%load target data

downsamplig_factor=3;  %CCC
n_str_tot=1; %number of structures  %CCC

path_to_data_indeces=['/home/cmau/PhD/MRI_data/UKBiobank/final_data/healthy_general/indeces_many_runs/'];  %CCC
path_to_data=['/dtu-compute/cmau/UKBiobank/final_data/healthy_general/T1/unbiased/nonlinear/final/down3/'];  %CCC
save_path='/home/cmau/PhD/RVoxM/latent_variables_variant/New_datasets/UKBiobank/healthy_general/new_linear/T1/many_runs/Ntrain2600/';


%%

 
show_figures=0;
show_imp_figures=0;

%n_latent_values=[ 0 200 400 500 600 700 800 900 1000 1300 1600 2000];

n_latent_values=[ 200];


n_fold_tot=numel(n_latent_values); %number of fold for cross validation

im_dim1=60;
im_dim2=72;
im_dim3=60;

 

%inizialization

n_it_max=500; %maximum number of iterations

num_round=1;
%voxels_all=cell(n_fold_tot,1);


num_it=zeros(num_round,n_fold_tot,n_str_tot);
V_matrix_all=cell(num_round,n_fold_tot,n_str_tot);
beta_all=cell(num_round,n_fold_tot,n_str_tot);
var_xn_posterior_all=zeros(num_round,n_fold_tot);
time_training=zeros(num_round,n_fold_tot);
time_test=zeros(num_round,n_fold_tot);

test_correlations=zeros(num_round,n_fold_tot);
test_MAEs=zeros(num_round,n_fold_tot);
test_RMSEs=zeros(num_round,n_fold_tot);
test_predictions=cell(num_round,n_fold_tot);

valid_correlations=zeros(num_round,n_fold_tot);
valid_MAEs=zeros(num_round,n_fold_tot);
valid_RMSEs=zeros(num_round,n_fold_tot);
logML_valid_all=zeros(num_round,n_fold_tot);
valid_predictions=cell(num_round,n_fold_tot);

train_correlations=zeros(num_round,n_fold_tot);
train_MAEs=zeros(num_round,n_fold_tot);
train_RMSEs=zeros(num_round,n_fold_tot);

logml_vector=zeros(num_round,n_fold_tot,n_it_max);

%test_predictions=cell(n_fold_tot,1);
%test_predictions_noprior=cell(n_fold_tot,1);
%train_predictions=cell(n_fold_tot,1);
%test_correlations=zeros(1,n_fold_tot);
%train_correlations=zeros(1,n_fold_tot);
%train_RMSEs=zeros(1,n_fold_tot);
%MSE=zeros(1,n_fold_tot);
%test_RMSEs=zeros(1,n_fold_tot);

%beta_vector=zeros(n_fold_tot,n_it_max);
%logml_vector=zeros(n_fold_tot,n_it_max);




%im_test_all=cell(n_fold_tot,1);
%im_train_all=cell(n_fold_tot,1);
%train_basis_fun_all=cell(1,n_fold_tot);
%test_basis_fun_all=cell(1,n_fold_tot);

%num_it=zeros(n_fold_tot,n_str_tot);

load([path_to_data,'age_uk_healthy_valid.mat']) %CCC
target_valid=age_healthy_valid(1:500);
 n_valid_images=size(target_valid,1);
covariates_valid=[];
clear('age_healthy_valid')

load([path_to_data,'age_uk_healthy_test.mat']) %CCC
test_target=age_healthy_test;
n_test_images=size(test_target,1);
covariates_test=[];
clear('age_healthy_test')





load([path_to_data_indeces,'indeces_2600train_b_all.mat'])
train_indeces=indeces_all(n_set,:)';
clear('indeces_all')


%load target and covariates    
    
load([path_to_data,'age_uk_healthy_train_b.mat']) %CCC

target_train=age_healthy(train_indeces);
N_train=size(target_train,1);
covariates_train=[];
clear('age_healthy')


    %train basis functions
   
    %train basis functions

train_basis_fun_init=[target_train,covariates_train];
     %matrix n_train_images x n_basis_fun with basis functions of all train subjects
 n_basis_fun=size(train_basis_fun_init,2); %number of basis functions
  

  train_basis_fun_mean=mean(train_basis_fun_init); %vector 1 x n_basis_fun with means
    train_basis_fun_std=std(train_basis_fun_init); %vector 1 x n_basis_fun with stds
    train_basis_fun=(train_basis_fun_init-train_basis_fun_mean)./train_basis_fun_std;  
 

   
%     train_covariates_init=train_basis_fun_init(:,2:end);
%     %matrix n_train_images x n_covariates with covariates of all train subjects
%     train_covariates_mean=mean(train_covariates_init); %vector 1 x n_basis_fun with means
%     train_covariates_std=std(train_covariates_init); %vector 1 x n_basis_fun with stds
%     
%     %train_covariates=train_covariates_init;
%     train_covariates=(train_covariates_init-train_covariates_mean)./train_covariates_std;    %CCC
%     %standardized train basis functions
    
    
   
    
    
    
    %test basis functions
    
    
    test_basis_fun=[test_target,covariates_test];
    st_test_basis_fun=(test_basis_fun-train_basis_fun_mean)./train_basis_fun_std;
    
    
   %valid basis functions
   
    valid_basis_fun=[target_valid,covariates_valid];
    st_valid_basis_fun=(valid_basis_fun-train_basis_fun_mean)./train_basis_fun_std;
    
    
    %loop over structures 
   % for str_n=1:n_str_tot
    
        %load images    
        %load(structure_names1{str_n})
        %load(structure_names2{str_n})
       % fprintf('structure n. = %d\n',str_n)
       % fprintf('\n')
        %[~,im_dim1,im_dim2,im_dim3]=size(MRI_oasis3_healthy_down2);   %CCC
%         [~,im_dim1,im_dim2,im_dim3]=size(MRI_IXI_down2_selected);   %CCC
%         vol_size=[im_dim1 im_dim2 im_dim3];
%         n_tot_voxel=vol_size(1)*vol_size(2)*vol_size(3);
%         %Xdata1=MRI_oasis3_healthy_down2;    %CCC
%         %clear('MRI_oasis3_healthy_down2')  %CCC
%         Xdata2=MRI_IXI_down2_selected;    %CCC
%         clear('MRI_IXI_down2_selected')  %CCC
%         %Xdata=[Xdata1; Xdata2];
%         Xdata=Xdata2;
%         Xdata=Xdata(perm,:,:,:);
        
        %select mask of voxels 
%         conc_th=conc_threshold(str_n); %concentration threshold
%         avg_vol= squeeze(mean(Xdata(rows_train,:,:,:),1)); %average volume      %CCC
%         %avg_vol= squeeze(mean(Xdata(:,:,:,rows_train),4)); %average volume
%         used_voxels = find(avg_vol(:)>conc_th)';  %row vector with flat indexes of voxels into the mask
%         n_voxels=numel(used_voxels); %number of voxels in the mask
%         avg_volume{n_fold}=avg_vol;

%load([path_to_data_train,'mask_indeces_train_healthy_2600.mat']) %load used_voxels
%load([path_to_data,'segm_uk_healthy_train_down3_b.mat']) %load im_train_init


%im_healthy_train=zeros(N_train,im_dim1,im_dim2,im_dim3);



load([path_to_data,'T1_nonlin_uk_healthy_train_b_down3.mat'])

im_healthy_train=T1_nonlin_down3(train_indeces,:,:,:);
clear('T1_nonlin_down3')



%win = centerCropWindow3d([91 109 91],[80 96 80]);
%cropped_images_train=im_healthy_train(:,7:86,8:103,7:86);
cropped_images_train=im_healthy_train(:,4:57,5:68,4:57);

clear('im_healthy_train')

im_train_init=zeros(N_train,54*54*64);
for n=1:N_train
    im_train_init(n,:)=reshape(cropped_images_train(n,:,:,:),54*54*64,1)';
end
 
 clear('croppped_images_train')




        im_train_mean=mean(im_train_init); %mean by column
        im_train_std=std(im_train_init);
        
        indeces_keep=find(im_train_mean>15);
        
       % indeces_keep=find(im_train_std>2);
       
%         mask_mni=niftiread('/home/cmau/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil1.nii.gz');
%          mask_cropped=mask_mni(7:86,8:103,7:86);
%          indeces_mask=find(mask_cropped==1);
%          loc_indeces=find(im_train_std(indeces_mask)>0);
%          indeces_keep=indeces_mask(loc_indeces);
         
         %%
%            I=zeros(80,96,80);
%         I(indeces_keep)=1;
%        
%       
%         v=reshape(im_train_init(40,:)',80,96,80);
%        
%         figure,
%         imagesc(imrotate(squeeze(I(40,:,:)),90)), colormap gray 
%           axis equal
%         figure,
%         imagesc(imrotate(squeeze(v(40,:,:)),90)), colormap gray 
%         axis equal
%         
%         figure,
%         imagesc(imrotate(squeeze(I(:,50,:)),90)), colormap gray 
%           axis equal
%         figure,
%         imagesc(imrotate(squeeze(v(:,50,:)),90)), colormap gray 
%         axis equal
%         
%         figure,
%         imagesc(imrotate(squeeze(I(:,:,40)),90)), colormap gray 
%           axis equal
%         figure,
%         imagesc(imrotate(squeeze(v(:,:,40)),90)), colormap gray 
%         axis equal
%         
%         figure,
%         imagesc(imrotate(squeeze(I(20,:,:)),90)), colormap gray 
%           axis equal
%         figure,
%         imagesc(imrotate(squeeze(v(20,:,:)),90)), colormap gray 
%         axis equal
%         
%         figure,
%         imagesc(imrotate(squeeze(I(:,30,:)),90)), colormap gray 
%           axis equal
%          figure,
%         imagesc(imrotate(squeeze(v(:,30,:)),90)), colormap gray 
%         axis equal
%         
%         figure,
%         imagesc(imrotate(squeeze(I(:,:,20)),90)), colormap gray
%           axis equal
%         figure,
%         imagesc(imrotate(squeeze(v(:,:,20)),90)), colormap gray
%         axis equal
%          
         
           
           %%
           
        im_train_init=im_train_init(:,indeces_keep);
        im_train_mean=im_train_mean(indeces_keep);
        im_train_std=im_train_std(indeces_keep);
        n_voxels=size(im_train_init,2); %number of voxels in the mask
        
       % im_train=im_train_init-im_train_mean; %matrix
         im_train=(im_train_init-im_train_mean)./im_train_std; %matrix
        clear('im_train_init')

        
        
       load([path_to_data,'T1_nonlin_uk_healthy_test_down3.mat']) %load T1_nonlin_down3

       cropped_images_test=T1_nonlin_down3(:,4:57,5:68,4:57);
clear('T1_nonlin_down3')

im_test=zeros(n_test_images,54*54*64);
for n=1:n_test_images
    im_test(n,:)=reshape(cropped_images_test(n,:,:,:),54*54*64,1)';
end
       clear('cropped_images_test')
       
       
         im_test=im_test(:,indeces_keep);
         st_im_test=(im_test-im_train_mean)./im_train_std;
   
        clear('im_test')
        
      load([path_to_data,'T1_nonlin_uk_healthy_valid_down3.mat']) %load T1_nonlin_down3

       cropped_images_valid=T1_nonlin_down3(1:500,4:57,5:68,4:57);
clear('T1_nonlin_down3')

im_valid=zeros(n_valid_images,54*54*64);
for n=1:n_valid_images
    im_valid(n,:)=reshape(cropped_images_valid(n,:,:,:),54*54*64,1)';
end
       
clear('cropped_images_valid')
        
        
          im_valid=im_valid(:,indeces_keep);
         st_im_valid=(im_valid-im_train_mean)./im_train_std;
   
       clear('im_valid')
         

          
           D_W=im_train'*train_basis_fun;
        A_W=train_basis_fun'*train_basis_fun;
        
        W_matrix=(A_W'\D_W')';
        w_star=W_matrix(:,1);
          
          
      
        for n_init=1:num_round
     
        
          %loop for latent_var
   for n_fold=1:n_fold_tot
        
       n_latent=n_latent_values(n_fold);
       fprintf('n.set = %d\n',n_set)
        fprintf('n.round = %d\n',n_init)
       fprintf('n.latent = %d\n',n_latent)
       
       
        %initialize parameters

        %beta_init=(1./(var(im_train)))'; %column vector
        
        beta_init=zeros(n_voxels,1);
        n_loop=ceil(n_voxels/20000);
        for i=1:n_loop
         indeces= 1+(i-1)*20000:min(i*20000,n_voxels);
         beta_init(indeces)=(1./(var(im_train(:,indeces))))'; %column vector
        end
           
        
        beta=beta_init;
        
       rng(n_init)
        
        %W_matrix_init=randn(n_voxels,n_basis_fun); %randomly sampled from standard Gaussian
        %W_matrix=W_matrix_init;
       % w=reshape(W_matrix',n_voxels*n_basis_fun,1);
        
        %v_init=
        %v=v_init;
        V_matrix_init=randn(n_voxels,n_latent); %randomly sampled from standard Gaussian;
        V_matrix=V_matrix_init;
       % v=reshape(V_matrix',n_voxels*n_latent,1);
        
        
        %mu=zeros(n_voxels*n_basis_fun,1);
        logML=inf;
        prev_logML=inf;
        
        
        cost_tol=1e-5; %tolerance for convergence of logml in the training
        %min_lambda=0.3;
        %max_lambda=1e5;
        %min_beta=1e-3;
        %max_beta=10^4;
        %min_alpha=1e-4;
        
        n_it_min=10;
        if n_latent==0
            n_it_min=3;
        end
      

        
        %mu_z=zeros(N_train,n_latent);
        %Sigma_zn_inv=cell(n_train_images(n_fold),1);
        %Sigma_zn=cell(n_train_images(n_fold),1);
        %Phi=kron(speye(n_voxels),train_basis_fun);
        
        bin_edges=1:n_basis_fun:n_voxels*n_basis_fun+1;
       
        voxel_per_batch=max(floor(1500/(n_basis_fun+n_latent)),1);
        %voxel_per_batch=1500;
        n_batches=ceil(n_voxels/voxel_per_batch); %n. of batches used for updates during training
        
        
        

        %start training
        tic
        for n_it = 1:n_it_max
            if (n_it > n_it_min)
                if (abs((logML - prev_logML)/logML) < cost_tol)   
                    
                   % w = prev_w;
                   % v= prev_v;
                   V_matrix=prev_V_matrix;
                   %W_matrix=prev_W_matrix;
                    beta = prev_beta;
                
                    break;
         
                end

            end
            fprintf('it number: %d\n', n_it)
    
            %save old values
            
            %prev_w = w;
          %  prev_v= v;
         % prev_W_matrix=W_matrix;
          prev_V_matrix=V_matrix;
            prev_beta = beta;
            prev_logML = logML;
            %prev_mu_z = mu_z;
    
            %save values for each iteration
           % beta_vector(n_fold,n_it)=mean(beta);
            %beta_vector2(n_fold,n_it)=median(beta);
       
             logml_vector(n_init,n_fold,n_it)=logML;
            
           % w_vector(n_fold,n_it)=mean(w);
           % w_vector2(n_fold,n_it)=median(w);
           % v_vector(n_fold,n_it)=mean(v); 
            %v_vector2(n_fold,n_it)=median(v);
            %mu_z_vector(n_fold,n_it)=mean(mu_z);
            %mu_z_vector2(n_fold,n_it)=median(mu_z); %anche Sigma_z?
   
        
            diag_betas=spdiags(beta,0,n_voxels,n_voxels);
            %B=kron(diag_betas,speye(N_train));
            %%
            
            %update of posterior mean and variance of latent variables
             Sigma_zn_inv=eye(n_latent)+V_matrix'*diag_betas*V_matrix;
             Sigma_zn=inv(Sigma_zn_inv);
             
%              tic
%              mu_z2=zeros(N_train,n_latent);
%             for n=1:N_train
%                 mu_z2(n,:)=(Sigma_zn_inv\(V_matrix'*diag_betas*(im_train(n,:)'- kron(speye(n_voxels),train_basis_fun(n,:))*w)))';
%             end
%             toc
            
            
            dim_block=8000;
            n_blocks_tot=ceil(N_train/dim_block);
            %summ_cell=cell(n_blocks_tot,1);
                     
            mu_z=zeros(N_train,n_latent);
            for n_block=1:n_blocks_tot
                indeces_block=(n_block-1)*dim_block+1:min(n_block*dim_block,N_train);
                Mat=im_train(indeces_block,:)-train_basis_fun(indeces_block,:)*W_matrix';
                mu_z(indeces_block,:)=Mat*diag_betas*V_matrix*Sigma_zn;
                
            end
            clear('Mat')
            
            
            %check which way is faster
            %VV=spalloc(n_train_images(n_fold)*n_voxels,n_latent*n_train_images(n_fold),n_latent*n_train_images(n_fold)*n_voxels);
            %for j=1:n_voxels
               %VV(1+(j-1)*n_train_images(n_fold):j*n_train_images(n_fold),:)= kron(speye(n_train_images(n_fold)),V_matrix(j,:));
            %end
            %Sigma_z_inv2=speye(n_train_images(n_fold)*n_latent)+VV'*B*VV;
            %mu_z2=Sigma_z_inv2\(VV'*B*(t-Phi*w));
            
            
            %compute log marginal likelihood
           
             
            %with Cholesky decomposition. For small number of latent
            %variables, there is no difference bewtween using cholesky and
            %computing determinants directly (in value and time), but if the number of latent
            %increases (e.g. 50) det is not robust (logML=-inf)
            %tic
            %log_det_Sigma_z_inv=zeros(N,1);
            %for n=1:N
                %U=chol(Sigma_zn_inv{n}); %detSigmaZninv=det(U)^2;
                %log_det_Sigma_z_inv(n)=2*sum(log(diag(U)));
            %end
            U=chol(Sigma_zn_inv); %detSigmaZninv=det(U)^2;
            log_det_Sigma_zn_inv=2*sum(log(diag(U)));
            %%
%             tic
%             logML2=-0.5*(-N_train*sum(log(beta))+N_train*log_det_Sigma_zn_inv...
%                 +(t-Phi*w)'* B *(t-Phi*w-reshape(mu_z*V_matrix',N_train*n_voxels,1)));
%             toc
%             fprintf('logML2: %.6e\n',logML2)
           %%
           % tic
            dim_block=1000;
            n_blocks_tot=ceil(N_train/dim_block);
            summ_cell=cell(n_blocks_tot,1);
                     
            
            for n_block=1:n_blocks_tot
                indeces_block=(n_block-1)*dim_block+1:min(n_block*dim_block,N_train);
                Mat=im_train(indeces_block,:)-train_basis_fun(indeces_block,:)*W_matrix';
                sum1=trace(Mat*diag_betas*Mat');
                sum2=trace( (Mat*diag_betas*V_matrix)*Sigma_zn*(V_matrix'*diag_betas*Mat'));
                summ_cell{n_block}=sum1-sum2;
            end
            clear('Mat')
            summ_logml=sum(cell2mat(summ_cell));
            
          logML=-0.5*(N_train*n_voxels*log(2*pi)-N_train*sum(log(beta))+N_train*log_det_Sigma_zn_inv+  summ_logml );
           % toc
            fprintf('logML: %.6e\n',logML)
    
    %%        
            
            
            
            
            
            %tic
            %det_Sigma_z_inv=zeros(N,1);
            %for n=1:N
                %det_Sigma_z_inv(n)=det(Sigma_z_inv{n});
            %end
            %logML2=-0.5*(-n_train_images(n_fold)*sum(log(beta))+sum(log(det_Sigma_z_inv))...
                %+(t-Phi*w)'* B *(t-Phi*w-reshape(mu_z*V_matrix',n_train_images(n_fold)*n_voxels,1)));
            %toc
            %fprintf('logML2: %.6e\n',logML2)
            
            
            %update of w,v
           % X=[train_basis_fun,mu_z];
           
           
%             
            
           A=mu_z'*mu_z+N_train.*Sigma_zn;
           D=im_train'*mu_z-W_matrix*train_basis_fun'*mu_z;
           V_matrix=transpose(A'\D');
            
            %w=reshape(W_matrix',n_voxels*n_basis_fun,1);
           % v=reshape(V_matrix',n_voxels*n_latent,1);
            
            %inizialize values for parallel loop
            beta_par=cell(n_batches,1);
            W_par=cell(n_batches,1);
            V_par=cell(n_batches,1);
            im_train_par=cell(1,n_batches);
           
            for i=1:n_batches
                voxel_ind=1+(i-1)*voxel_per_batch:min(n_voxels,i*voxel_per_batch);%indixes of voxels of current batch
               
                %beta_par{i}=beta(voxel_ind); %previous betas of the batch
                im_train_par{i}=im_train(:,voxel_ind); %train images of the batch (voxels in the batch)
                W_par{i}=W_matrix(voxel_ind,:);
                V_par{i}=V_matrix(voxel_ind,:);
                
            end
            
            
   
            %parallel loop for computing betas
            parfor i=1:n_batches
               
                %compute elements of beta corresponding to the batch
                
                 beta_par{i}=N_train./...
                    ((sum((im_train_par{i}-train_basis_fun*W_par{i}'-...
                    mu_z*V_par{i}').^2))'+diag(V_par{i}*N_train*Sigma_zn*V_par{i}'));
                
            end
            

            %update of betas
            
            beta=cell2mat(beta_par); 
            
    
    
        end %end of loop over iterations
        time_training(n_init,n_fold)=toc;

  %% 

        fprintf('number of iterations: %d\n', n_it-1)
        fprintf('\n')

        %save interesting values
        str_n=1;
        
       % w=reshape(W_matrix',n_voxels*n_basis_fun,1);
       % v=reshape(V_matrix',n_voxels*n_latent,1);
        
        
        num_it(n_init,n_fold,str_n)=n_it-1;
       % w_all{n_fold,str_n}=w;
       % v_all{n_fold,str_n}=v;
        
        %W_matrix_all{str_n}=W_matrix;
        V_matrix_all{n_init,n_fold,str_n}=V_matrix;
        
        beta_all{n_init,n_fold,str_n}=beta;
       % mu_z_all{n_init,n_fold,str_n}=mu_z;
        %Sigma_z_all{n_init,n_fold,str_n}=Sigma_zn;
        
       % im_train_all{n_fold,str_n}=im_train;
      %  voxels_all{n_fold,str_n}=used_voxels;
       % n_voxels_all(n_fold,str_n)=numel(used_voxels);
        %im_test_st_all{n_fold,str_n}=st_im_test;
       % st_test_basis_fun_all{n_set}=st_test_basis_fun;
       % test_basis_fun_all{n_fold}=test_basis_fun;
        %train_basis_fun_mean_all{n_fold}=train_basis_fun_mean;
        %train_basis_fun_std_all{n_fold}=train_basis_fun_std;
        %im_train_mean_all{n_fold,str_n}=im_train_mean;
        
        %im_valid_st_all{n_fold,str_n}=st_im_valid;
        %st_valid_basis_fun_all{n_set}=st_valid_basis_fun;
        

        %% plots

        %plot of different values through iterations
         if show_imp_figures   
            figure
            plot(1:num_it(n_fold,str_n),beta_vector(n_fold,1:num_it(n_fold,str_n)))
            xlabel('iteration number')
            title('\beta')
         
            figure
            plot(1:num_it(n_fold,str_n)-1,logml_vector(n_fold,2:num_it(n_fold,str_n)))
            xlabel('iteration number')
            title('logML')

         end


%% forward predicitons


    %forward predictions test
%     [test_pred_mean,test_error_mean_tot,test_error_mean_rel,test_error_all_tot,test_error_all_rel,test_cor_mean_tot,...
%     test_cor_mean_rel]...
%     =forward_predictions(mu_all{n_fold,str_n},st_test_basis_fun_all{n_fold},im_test_st_all{n_fold,str_n}+im_train_mean_all{n_fold,str_n},im_train_mean_all{n_fold,str_n},relevant_indexes_all{n_fold,str_n},relevant_voxels_all{n_fold,str_n},n_voxels_all(n_fold,str_n));
% 
% 
%     %forward predictions train
%     [train_pred_mean,train_error_mean_tot,train_error_mean_rel,train_error_all_tot,train_error_all_rel,train_cor_mean_tot,...
%     train_cor_mean_rel]...
%     =forward_predictions(mu_all{n_fold,str_n},train_basis_fun_all{n_fold},im_train_all{n_fold,str_n}+im_train_mean_all{n_fold,str_n},im_train_mean_all{n_fold,str_n},relevant_indexes_all{n_fold,str_n},relevant_voxels_all{n_fold,str_n},n_voxels_all(n_fold,str_n));
% 
%     fprintf('Forward predicitons - test performances\n')
%     fprintf('mean test error (all voxels): %.4f\n', test_error_mean_tot)
%     fprintf('mean test error (relevant voxels): %.4f\n', test_error_mean_rel)
%     fprintf('total test error (all voxels): %.4f\n', test_error_all_tot)
%     fprintf('total test error (relevant voxels): %.4f\n', test_error_all_rel)
%     fprintf('mean test correlation (all voxels): %.4f\n', test_cor_mean_tot)
%     fprintf('mean test correlation (relevant voxels): %.4f\n', test_cor_mean_rel)
%     fprintf('\n')
% 
% 
%     fprintf('Forward predicitons - train performances\n')
%     fprintf('mean train error (all voxels): %.4f\n', train_error_mean_tot)
%     fprintf('mean train error (relevant voxels): %.4f\n', train_error_mean_rel)
%     fprintf('total train error (all voxels): %.4f\n', train_error_all_tot)
%     fprintf('total train error (relevant voxels): %.4f\n', train_error_all_rel)
%     fprintf('mean train correlation (all voxels): %.4f\n', train_cor_mean_tot)
%     fprintf('mean train correlation (relevant voxels): %.4f\n', train_cor_mean_rel)
%     fprintf('\n')
 
    %% visualize predicted images
    
    if show_figures
    

        % train - predicted images

        I=zeros(n_tot_voxel,1);
        us_vox=voxels_all{n_fold,str_n};
        j=44; %subject number
        %I(us_vox(relevant_voxels))=predictions{k}(:,j);
        I(us_vox)=train_pred_mean(:,j);
        imag=reshape(I,im_dim1,im_dim2,im_dim3);
        I2=zeros(n_tot_voxel,1);
        I2(us_vox)=(im_train_all{n_fold,str_n}(j,:)+im_train_mean_all{n_fold,str_n})';
        imag_real=reshape(I2,im_dim1,im_dim2,im_dim3);

        %for i=10:10:im_dim1-10
        i=35; %for left cortex
        %i=20;  %for right cortex
        %figure, imagesc(imrotate(squeeze(avg_vol(i,:,:)),90)), colormap gray
        figure, imagesc(imrotate(squeeze(imag(i,:,:)),0)), colormap gray, colorbar
        figure, imagesc(imrotate(squeeze(imag_real(i,:,:)),0)), colormap gray, colorbar
        %end


        %for i=10:10:im_dim2-10
        i=25;
        %figure, imagesc(imrotate(squeeze(avg_vol(i,:,:)),90)), colormap gray
        figure, imagesc(imrotate(squeeze(imag(:,i,:)),-90)), colormap gray, colorbar
        figure, imagesc(imrotate(squeeze(imag_real(:,i,:)),-90)), colormap gray, colorbar
        %end

        %for i=10:10:im_dim3-10
        i=50;
        %figure, imagesc(imrotate(squeeze(avg_vol(i,:,:)),90)), colormap gray
        figure, imagesc(imrotate(squeeze(imag(:,:,i)),-90)), colormap gray, colorbar
        figure, imagesc(imrotate(squeeze(imag_real(:,:,i)),-90)), colormap gray, colorbar
        %end

        figure, scatter(I2(us_vox),train_pred_mean(:,j))

        % test - predicted images

        I=zeros(n_tot_voxel,1);
        us_vox=voxels_all{n_fold,str_n};
        j=15; %subject number
        I(us_vox)=test_pred_mean(:,j);
        imag=reshape(I,im_dim1,im_dim2,im_dim3);
        I2=zeros(n_tot_voxel,1);
        I2(us_vox)=(im_test_st_all{n_fold,str_n}(j,:)+im_train_mean_all{n_fold,str_n})';
        imag_real=reshape(I2,im_dim1,im_dim2,im_dim3);

        %for i=10:10:im_dim1-10
        i=35; %for left cortex
        %i=20; %for right ocrtex
        %figure, imagesc(imrotate(squeeze(avg_vol(i,:,:)),90)), colormap gray
        figure, imagesc(imrotate(squeeze(imag(i,:,:)),0)), colormap gray, colorbar
        figure, imagesc(imrotate(squeeze(imag_real(i,:,:)),0)), colormap gray, colorbar
        %end


        %for i=10:10:im_dim2-10
        i=25;
        %figure, imagesc(imrotate(squeeze(avg_vol(i,:,:)),90)), colormap gray
        figure, imagesc(imrotate(squeeze(imag(:,i,:)),-90)), colormap gray, colorbar
        figure, imagesc(imrotate(squeeze(imag_real(:,i,:)),-90)), colormap gray, colorbar
        %end

        %for i=10:10:im_dim3-10
        i=50;
        %figure, imagesc(imrotate(squeeze(avg_vol(i,:,:)),90)), colormap gray
        figure, imagesc(imrotate(squeeze(imag(:,:,i)),-90)), colormap gray, colorbar
        figure, imagesc(imrotate(squeeze(imag_real(:,:,i)),-90)), colormap gray, colorbar
        %end

        figure, scatter(I2(us_vox),test_pred_mean(:,j))
 

    end
    
   % end %end loop over structures

    
%% target predictions

   

    target_train_mean=train_basis_fun_mean(1);
    target_train_std=train_basis_fun_std(1);
    
    var_x_prior=1;
    diag_betas=spdiags(beta,0,n_voxels,n_voxels);
    Sigma_zn_inv=eye(n_latent)+V_matrix'*diag_betas*V_matrix;
    Sigma_zn=inv(Sigma_zn_inv);
    
   
    tic
    Delta_times_wstar=diag_betas*w_star;
    aux=V_matrix'*Delta_times_wstar;
    var_xn_posterior=1/(1/var_x_prior+w_star'*Delta_times_wstar-aux'*Sigma_zn*aux);
    var_xn_posterior_all(n_init,n_fold)=var_xn_posterior;
    
    %test
%     tic
%     mean_xn_test=zeros(n_test_images,1); %column vector
%     for n=1:n_test_images
%         gap=st_im_test(n,:)'-W_matrix(:,2:end)*st_test_basis_fun(n,2:end)';
%         aux2=V_matrix'*diag_betas*gap;
%         mean_xn_test(n)=var_xn_posterior*(Delta_times_wstar'*gap-aux'*Sigma_zn*aux2);
%         
%     end
%     toc
    
   
     gap_all=st_im_test'-W_matrix(:,2:end)*st_test_basis_fun(:,2:end)';
    aux2_all=V_matrix'*diag_betas*gap_all;
    mean_xn_test=var_xn_posterior.*(Delta_times_wstar'*gap_all-aux'*Sigma_zn*aux2_all)';
    time_test(n_init,n_fold)=toc;
    
    %take predictions back to the original target space
    test_pred_target=(mean_xn_test.*target_train_std+target_train_mean);  %CCC
    
    test_correlation=corr(test_target,test_pred_target);
    test_RMSE=sqrt(sum((test_target-test_pred_target).^2)/n_test_images);
    test_MAE=mean(abs(test_target-test_pred_target));
   
    
  

    %compute correlation and error
  
    test_correlations(n_init,n_fold)=test_correlation;
    test_RMSEs(n_init,n_fold)=test_RMSE;
    test_MAEs(n_init,n_fold)=test_MAE;
    test_predictions{n_init,n_fold}=single(test_pred_target);

    fprintf('test correlation without prior: %.4f\n',test_correlation)
    fprintf('test RMSE without prior: %.4f\n',test_RMSE)
    fprintf('test MAE without prior: %.4f\n',test_MAE)
    fprintf('\n')
%%
    if show_imp_figures
        %%
        figure, scatter(test_target,test_pred_target)
        hold on
        plot(test_target,test_target)
        xlabel('true score')
        ylabel('predicited score')
        title('test predictions')
        hold off
    end

%%  visualize posterior distribution 
    
if show_figures
    real_test_target_st=(target(rows_test_all{n_fold})-target_train_mean)./target_train_std;
    for j=1:2:n_test_images(n_fold)
    figure,
    plot( bin_centers,test_distribution(j,:))
    hold on
    plot(test_pred_target_st(j),0,'r*')
    plot(real_test_target_st(j),0,'b*')
    legend('posterior','mean','real target')
    title(['subject', num2str(j)])
     end

    for j=1:2:n_test_images(n_fold)
    figure,
    plot( bin_centers,test_distribution_noprior(j,:))
    hold on
    plot(st_test_pred_target_noprior(j),0,'r*')
    plot(real_test_target_st(j),0,'b*')
    legend('posterior','mean','real target')
    title(['subject', num2str(j),' no prior'])
    end
end

%valid target predictions

    
%      mean_xn_valid=zeros(n_valid_images,1); %column vector
%     for n=1:n_valid_images
%         gap=st_im_valid(n,:)'-W_matrix(:,2:end)*st_valid_basis_fun(n,2:end)';
%         aux2=V_matrix'*diag_betas*gap;
%         mean_xn_valid(n)=var_xn_posterior*(Delta_times_wstar'*gap-aux'*Sigma_zn*aux2);
%         
%     end
     
     gap_all=st_im_valid'-W_matrix(:,2:end)*st_valid_basis_fun(:,2:end)';
    aux2_all=V_matrix'*diag_betas*gap_all;
    mean_xn_valid=var_xn_posterior.*(Delta_times_wstar'*gap_all-aux'*Sigma_zn*aux2_all)';
    
    
    %take predictions back to the original target space
    valid_pred_target=(mean_xn_valid.*target_train_std+target_train_mean);  %CCC
          
       %compute correlation and error
    valid_correlation=corr(target_valid,valid_pred_target);
    valid_RMSE=sqrt(sum((target_valid-valid_pred_target).^2)/n_valid_images);
    valid_MAE=mean(abs(target_valid-valid_pred_target));

    
    valid_correlations(n_init,n_fold)=valid_correlation;
    valid_RMSEs(n_init,n_fold)=valid_RMSE;
    valid_MAEs(n_init,n_fold)=valid_MAE;
    valid_predictions{n_init,n_fold}=valid_pred_target;

    fprintf('valid correlation without prior: %.4f\n',valid_correlation)
    fprintf('valid RMSE without prior: %.4f\n',valid_RMSE)
    fprintf('valid MAE without prior: %.4f\n',valid_MAE)
    fprintf('\n')

    if show_imp_figures
        figure, scatter(target_valid,valid_pred_target)
        hold on
        plot(target_valid,target_valid)
        xlabel('true score')
        ylabel('predicited score')
        title('valid predictions')
        hold off
    end


%      mean_xn_train=zeros(N_train,1); %column vector
%     for n=1:N_train
%         gap=im_train(n,:)'-W_matrix(:,2:end)*train_basis_fun(n,2:end)';
%         aux2=V_matrix'*diag_betas*gap;
%         mean_xn_train(n)=var_xn_posterior*(Delta_times_wstar'*gap-aux'*Sigma_zn*aux2);
%         
%     end

    gap_all=im_train'-W_matrix(:,2:end)*train_basis_fun(:,2:end)';
    aux2_all=V_matrix'*diag_betas*gap_all;
    mean_xn_train=var_xn_posterior.*(Delta_times_wstar'*gap_all-aux'*Sigma_zn*aux2_all)';
    
    
     
    
    %take predictions back to the original target space
    train_pred_target=(mean_xn_train.*target_train_std+target_train_mean);  %CCC

 train_correlation=corr(target_train,train_pred_target);
    train_RMSE=sqrt(sum((target_train-train_pred_target).^2)/N_train);
   train_MAE=mean(abs(target_train-train_pred_target));
   

train_correlations(n_init,n_fold)=train_correlation;
    train_RMSEs(n_init,n_fold)=train_RMSE;
    train_MAEs(n_init,n_fold)=train_MAE;
    %train_predictions{n_init,n_fold}=train_pred_target;



    fprintf('train correlation without prior: %.4f\n',train_correlation)
    fprintf('train RMSE without prior: %.4f\n',train_RMSE)
    fprintf('train MAE without prior: %.4f\n',train_MAE)
    fprintf('\n')
    %%
    if show_imp_figures
        figure, scatter(target_train,train_pred_target)
        hold on
        plot(target_train,target_train)
        xlabel('true score')
        ylabel('predicited score')
        title('train predictions')
        hold off
    end
    
%% logML

             
           
U=chol(Sigma_zn_inv); %detSigmaZninv=det(U)^2;
log_det_Sigma_zn_inv=2*sum(log(diag(U)));

dim_block=1000;
n_blocks_tot=ceil(n_valid_images/dim_block);
summ_cell=cell(n_blocks_tot,1);
                     
for n_block=1:n_blocks_tot
    indeces_block=(n_block-1)*dim_block+1:min(n_block*dim_block,n_valid_images);
    Mat=st_im_valid(indeces_block,:)-st_valid_basis_fun(indeces_block,:)*W_matrix';
    sum1=trace(Mat*diag_betas*Mat');
    sum2=trace( (Mat*diag_betas*V_matrix)*Sigma_zn*(V_matrix'*diag_betas*Mat'));
    summ_cell{n_block}=sum1-sum2;
end
clear('Mat')
summ_logml=sum(cell2mat(summ_cell));
            
logML_valid=-0.5*(-n_valid_images*sum(log(beta))+n_valid_images*log_det_Sigma_zn_inv+summ_logml );
           
 fprintf('logML valid: %.6e\n',logML_valid)
 logML_valid_all(n_init,n_fold)=logML_valid;
 
 
% dim_block=1000;
% n_blocks_tot=ceil(n_test_images/dim_block);
% summ_cell=cell(n_blocks_tot,1);
%                      
% for n_block=1:n_blocks_tot
%     indeces_block=(n_block-1)*dim_block+1:min(n_block*dim_block,n_test_images);
%     Mat=st_im_test(indeces_block,:)-st_test_basis_fun(indeces_block,:)*W_matrix';
%     sum1=trace(Mat*diag_betas*Mat');
%     sum2=trace( (Mat*diag_betas*V_matrix)*Sigma_zn*(V_matrix'*diag_betas*Mat'));
%     summ_cell{n_block}=sum1-sum2;
% end
% clear('Mat')
% summ_logml=sum(cell2mat(summ_cell));
%             
% logML_test=-0.5*(-n_test_images*sum(log(beta))+n_test_images*log_det_Sigma_zn_inv+summ_logml);
%            
%  fprintf('logML test: %.6e\n',logML_test)
%  logML_test_all(n_init,n_fold)=logML_test;

 
%  dim_block=1000;
% n_blocks_tot=ceil(N_train/dim_block);
% summ_cell=cell(n_blocks_tot,1);
%                      
% for n_block=1:n_blocks_tot
%     indeces_block=(n_block-1)*dim_block+1:min(n_block*dim_block,N_train);
%     Mat=im_train(indeces_block,:)-train_basis_fun(indeces_block,:)*W_matrix';
%     sum1=trace(Mat*diag_betas*Mat');
%     sum2=trace( (Mat*diag_betas*V_matrix)*Sigma_zn*(V_matrix'*diag_betas*Mat'));
%     summ_cell{n_block}=sum1-sum2;
% end
% clear('Mat')
% summ_logml=sum(cell2mat(summ_cell));
%             
% logML_train=-0.5*(-N_train*sum(log(beta))+N_train*log_det_Sigma_zn_inv+summ_logml);
%            
%  fprintf('logML train: %.6e\n',logML_train)
%  logML_train_all(n_init,n_fold)=logML_train;

res.W_matrix=single(W_matrix);
res.V_matrix=single(V_matrix);
res.beta=single(beta);
res.indeces_keep=indeces_keep;


res.n_latent=n_latent_values(n_fold);
res.num_it=n_it-1;


res.train_basis_fun_mean=train_basis_fun_mean;
res.train_basis_fun_std=train_basis_fun_std;
res.im_train_mean=im_train_mean;
res.im_train_std=im_train_std;

%res.avg_vol=avg_vol;


res.valid_correlation=valid_correlation;
 res.valid_MAE= valid_MAE;
  res.valid_RMSE= valid_RMSE;
  
  res.test_correlation=test_correlation;
 res.test_MAE= test_MAE;
  res.test_RMSE= test_RMSE;
  
% res.train_correlation=train_correlation;
%  res.train_MAE= train_MAE;
%   res.train_RMSE= train_RMSE;
  
  
  res.logML_valid=logML_valid;
  %res.logML_test_all=logML_test_all;
  %res.logML_train_all=logML_train_all;
  
  
   res.test_predictions=single(test_pred_target);
  res.valid_predictions=single(valid_pred_target);
  %res.train_predictions=train_predictions;
  
  res.time_training=time_training(n_init,n_fold);
  res.time_test=time_test(n_init,n_fold);
  %res.logml_vector=logml_vector;
  
  save([save_path,'uk_T1_nonlin_2600train_down3_loop_set_b_set',num2str(n_set),'_',num2str(n_latent),'latent_new_logml.mat'],'res','-v7.3')
    clear('res')
    
   end
  end
   %%
    fprintf('\n')
    %fprintf('test correlations: %f \n',test_correlations_noprior)
   valid_correlations
   valid_MAEs
   valid_RMSEs
   
    test_correlations
   test_MAEs
   test_RMSEs
   
    train_correlations
   train_MAEs
   train_RMSEs
   
   logML_valid_all
   %logML_test_all
   %logML_train_all
   
   n_latent_values
    
   
   %%

%    for i=1:num_round
%        for j=1:n_fold_tot
%           % for i=1:3
%        %for j=1:32
%         for k=1:1
%        V_matrix_all{i,j,k}= single(V_matrix_all{i,j,k});
%        beta_all{i,j,k}=single(beta_all{i,j,k});
%         end
%        end
%    end
%    
%   
%         
%        W_matrix= single(W_matrix);
%        
%       
% %% save results
% 
% res.W_matrix=W_matrix;
% res.V_matrix_all=V_matrix_all;
% res.beta_all=beta_all;
% res.indeces_keep=indeces_keep;
% %res.mu_z_all=mu_z_all;
% %res.Sigma_z_all=Sigma_z_all;
% 
% res.n_latent=n_latent_values;
% %res.logml_vector=logml_vector;
% res.num_it=num_it;
% %res.beta_vector=beta_vector;
% 
% 
% res.train_basis_fun_mean=train_basis_fun_mean;
% res.train_basis_fun_std=train_basis_fun_std;
% res.im_train_mean=im_train_mean;
% res.im_train_std=im_train_std;
% 
% %res.avg_vol=avg_vol;
% 
% 
% res.valid_correlations=valid_correlations;
%  res.valid_MAEs= valid_MAEs;
%   res.valid_RMSEs= valid_RMSEs;
%   
%   res.test_correlations=test_correlations;
%  res.test_MAEs= test_MAEs;
%   res.test_RMSEs= test_RMSEs;
%   
% res.train_correlations=train_correlations;
%  res.train_MAEs= train_MAEs;
%   res.train_RMSEs= train_RMSEs;
%   
%   
%   res.logML_valid_all=logML_valid_all;
%   %res.logML_test_all=logML_test_all;
%   %res.logML_train_all=logML_train_all;
%   %%
%   
%    res.test_predictions=test_predictions;
%   res.valid_predictions=valid_predictions;
%   %res.train_predictions=train_predictions;
%   
%   res.time_training=time_training;
%   res.time_test=time_test;
%   res.logml_vector=logml_vector;
  
%%
%save([save_path,'uk_T1_nonlin_2600train_down3_loop_set_b_set',num2str(n_set),'_new_logml.mat'],'res','-v7.3')

%res.V_matrix_all=[];
%res.beta_all=[];

%save([save_path,'res_uk_T1_nonlin_2600train_down3_loop_set_b_set',num2str(n_set),'_new_logml.mat'],'res','-v7.3')

clear all

end

