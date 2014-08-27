void new_func2() {
    
}
void new_func() {
    //hey i will put my new codes here.
}
//average SGD implemented by Julius 2014.08.27
void LLC_SGD(double *w, double *x, double *centers, int *knn_idx, int knn, int d) {
    int i, j, iter=0, iter0=-1;
    float GAMMA = 2, adaGAMMA=GAMMA;
    memset(w, 0.01, knn*sizeof(double));
    
    //minimizing 0.5*||x - w*centers||2,2 + 0.5*lambda*||w||2,2
    srand(time(NULL));
    float past_grad[64] = {0};

    //w[0] = w[1] = 0.25;
    
    init_w(w, knn);
    eval_obj_iter = 0;
    while(iter < knn*30) {
        for(i=0; i<BATCH_SIZE; i++)
            batch_idx[i] = rand()%d;    //[0, 127]
        
        adaGAMMA = GAMMA*pow(1+GAMMA*BETA_SGD*(iter0+1), -0.75);
        //adaGAMMA = GAMMA;
        if(iter > knn*8)
            iter0++;
        //if(iter0 == 0)
        //    printf("============Start averaging!!============\n");
        
        //printf("adaGAMMA[%d]: %f\n", iter, adaGAMMA);
        
        float wcenters[BATCH_SIZE]={0}, sub_grad=0;
        
        //mini-batch SGD, if BATCH_SIZE = 1, then it is shrunk to SGD
        
        if(BATCH_SIZE > 1) {    //mini-batch SGD
            for(j=0; j<BATCH_SIZE; j++) {
                int idx = batch_idx[j];
                for(i=0; i<knn; i++)
                    wcenters[j] += w[i]*centers[knn_idx[i]*d+idx];
                wcenters[j] = x[idx] - wcenters[j];
            }
            
            for(i=0; i<knn; i++) {
                sub_grad=0;
                for(j=0; j<BATCH_SIZE; j++) {
                    int idx = batch_idx[j];
                    sub_grad += wcenters[j]*centers[knn_idx[i]*d+idx];
                }
                
                sub_grad /= BATCH_SIZE; //average over mini-batch gradients
                
                sub_grad += BETA_SGD*w[i];
                sub_grad *= adaGAMMA;
                
                if(abs_float(sub_grad) < 5e-4)  //neglect too small update
                    continue;
                
                w[i] += (sub_grad);
                
                w[i] = (w[i] > 1) ? 1: w[i];
                w[i] = (w[i] < -1) ? -1: w[i];
            }
        }
        else {  //SGD
            int idx = batch_idx[0];
            for(i=0; i<knn; i++)
                wcenters[0] += w[i]*centers[knn_idx[i]*d+idx];
            for(i=0; i<knn; i++) {
                sub_grad = (x[idx] - wcenters[0])*centers[knn_idx[i]*d+idx];    //calculate sub_grad[0~127]
                sub_grad += BETA_SGD*w[i];
                sub_grad *= adaGAMMA;
                
                if(abs_float(sub_grad) < 5e-4)  //neglect too small update
                    continue;
                
                double lower_bound, upper_bound;
                double tentaive_w = w[i] + sub_grad;
                if(w[i] > 0) {
                    lower_bound = 0;  //avg_w - 2*avg < x < avg_w + 2*avg
                    upper_bound = 2*w[i];
                }
                else {
                    lower_bound = 2*w[i];
                    upper_bound = 0;
                }
                
                w[i] += (sub_grad);
                
                w[i] = (tentaive_w > upper_bound) ? upper_bound: tentaive_w;
                w[i] = (tentaive_w < lower_bound) ? lower_bound: tentaive_w;
                
                w[i] = (w[i] > 1) ? 1: w[i];
                w[i] = (w[i] < -1) ? -1: w[i];
            }
        }
        
        if(iter0 == 0) {
            for(i=0; i<knn; i++)
                avg_w[i] = w[i];
        }
        if(iter0 >= 0) {
            for(i=0; i<knn; i++) {
                double past_w = avg_w[i];
                avg_w[i] = w[i] + (double)iter0/(iter0+1)*(avg_w[i]-w[i]);
                past_grad[i] = avg_w[i] - past_w;
                
                double tentaive_w = avg_w[i] + 2*past_grad[i];  //momentum is used
                
                double lower_bound, upper_bound;
                if(avg_w[i] > 0) {
                    lower_bound = -2*avg_w[i];  //avg_w - 3*avg < x < avg_w + 3*avg
                    upper_bound = 4*avg_w[i];
                }
                else {
                    lower_bound = 4*avg_w[i];
                    upper_bound = -2*avg_w[i];
                }
                avg_w[i] = (tentaive_w > upper_bound) ? upper_bound: tentaive_w;
                avg_w[i] = (tentaive_w < lower_bound) ? lower_bound: tentaive_w;
                
                avg_w[i] = (avg_w[i] > 1) ? 1: avg_w[i];
                avg_w[i] = (avg_w[i] < -1) ? -1: avg_w[i];
                
                //printf("[%d](w, grad) = (%f, %f)\n", i, avg_w[i], past_grad[i]);
            }
#ifdef SGD_DEBUGGING
            eval_obj(avg_w, x, centers, knn_idx, knn, d);
#endif
        }
        //norm_w(w, d);
        else {
#ifdef SGD_DEBUGGING
            eval_obj(w, x, centers, knn_idx, knn, d);
#endif
        }
        
        iter++;
    }
    double sum=0;
    for(i=0; i<knn; i++)
        sum += avg_w[i];
    
    for(i=0; i<knn; i++) {
        avg_w[i] /= sum;
        if(abs_double(avg_w[i]) < TOLERANCE)    //cut-off small values
            avg_w[i] = 0;
    }
    
#ifdef SGD_DEBUGGING
    printf("LLC_SGD w: (");
    for(i=0; i<knn; i++)
        printf("%f ", avg_w[i]);
    printf(")\n");
#endif
}