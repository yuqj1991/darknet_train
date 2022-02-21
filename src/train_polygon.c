#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "demo.h"
#include "option_list.h"

#include <sys/stat.h>
#include<stdio.h>
#include<time.h>
#include<sys/types.h>

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

#include "http_stream.h"

int check_mistakes_poly = 0;
#define FILEPATH_MAX (80)

float validate_polyGon_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, 
                            const float iou_thresh, const int map_points, int letter_box, 
                            network *existing_net) {
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size);
    FILE* reinforcement_fd = NULL;

    network net;
    //int initial_batch;
    if (existing_net) {
        char *train_images = option_find_str(options, "train", "data/train.txt");
        valid_images = option_find_str(options, "valid", train_images);
        net = *existing_net;
        remember_network_recurrent_state(*existing_net);
        free_network_recurrent_state(*existing_net);
    }
    else {
        net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
    }
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        getchar();
    }
    srand(time(0));
    printf("\n calculation polyGon (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    char **paths_dif = NULL;
    if (difficult_valid_images) {
        list *plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }


    layer l = net.layers[net.n - 1];
    int classes = l.classes;

    int m = plist->size;
    int i = 0;
    int t;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.type = HOMEPAGE_DATA;

    float mean_average_precision = 0.;
    int label_count = 0;
    int false_matched = 0;
    int missing_matched = 0;
    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "\r%d, nthreads: %d, m: %d", i, nthreads, m);
        for (t = 0; t < nthreads && (i + t - nthreads) < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && (i + t) < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);

            PolyGon_S *dets = (PolyGon_S*)xcalloc(2, sizeof(PolyGon_S));
            
            get_poly_result(net.layers[net.n - 1].output, dets, 0.5);
                       
            char labelpath[4096];
            replace_image_to_label(path, labelpath);
            int num_labels = 0;
            PolyGon_S *truth = fill_polyGon_labels(labelpath, &num_labels);
            int j;
            for (j = 0; j < 2; ++j) {
                // 只有左页或者右页，而预测出来为多出来的，
                if ((truth[j].label_id == -1) && (dets[j].label_id != -1)) {
                    false_matched ++;
                } else if ((truth[j].label_id != -1) && (dets[j].label_id == -1)) {
                    missing_matched++;
                } else if ((truth[j].label_id = -1) && (dets[j].label_id == -1)) {
                    continue;
                }
                if (dets[j].label_id == truth[j].label_id) {
                    float points_diff = 0;
                    mean_average_precision += pow(dets[j].leftTop.x - truth[j].leftTop.x, 2);
                    mean_average_precision += pow(dets[j].leftTop.y - truth[j].leftTop.y, 2);
                    mean_average_precision += pow(dets[j].rightTop.x - truth[j].rightTop.x, 2);
                    mean_average_precision += pow(dets[j].rightTop.y - truth[j].rightTop.y, 2);
                    mean_average_precision += pow(dets[j].rightBottom.x - truth[j].rightBottom.x, 2);
                    mean_average_precision += pow(dets[j].rightBottom.y - truth[j].rightBottom.y, 2);
                    mean_average_precision += pow(dets[j].leftBottom.x - truth[j].leftBottom.x, 2);
                    mean_average_precision += pow(dets[j].leftBottom.y - truth[j].leftBottom.y, 2);
                    label_count++;
                }
            }
            free(dets);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }

    return mean_average_precision / label_count;
}


void train_polyGon(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, 
                    int clear, int dont_show, int calc_map, int mjpeg_port, int show_imgs, 
                    int benchmark_layers, char* chart_path)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *valid_images = option_find_str(options, "valid", train_images);
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    network net_map;
    if (calc_map) {
        FILE* valid_file = fopen(valid_images, "r");
        if (!valid_file) {
            printf("\n Error: There is no %s file for mAP calculation!\n Don't use -map flag.\n Or set valid=%s in your %s file. \n", valid_images, train_images, datacfg);
            getchar();
            exit(-1);
        }
        else fclose(valid_file);

        cuda_set_device(gpus[0]);
        printf(" Prepare additional network for mAP calculation...\n");
        net_map = parse_network_cfg_custom(cfgfile, 1, 1);
        net_map.benchmark_layers = benchmark_layers;
        const int net_classes = net_map.layers[net_map.n - 1].classes;

        int k;  // free memory unnecessary arrays
        for (k = 0; k < net_map.n - 1; ++k) free_layer_custom(net_map.layers[k], 1);

        char *name_list = option_find_str(options, "names", "data/names.list");
        int names_size = 0;
        char **names = get_labels_custom(name_list, &names_size);
        if (net_classes != names_size) {
            printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
                name_list, names_size, net_classes, cfgfile);
            if (net_classes > names_size) getchar();
        }
        free_ptrs((void**)names, net_map.layers[net_map.n - 1].classes);
    }

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network* nets = (network*)xcalloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int k;
    for (k = 0; k < ngpus; ++k) {
        printf("gpus[k]: %d\n", gpus[k]);
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[k]);
#endif
        nets[k] = parse_network_cfg(cfgfile);
        nets[k].benchmark_layers = benchmark_layers;
        if (weightfile) {
            load_weights(&nets[k], weightfile);
        }
        if (clear) {
            *nets[k].seen = 0;
            *nets[k].cur_iteration = 0;
        }
        nets[k].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    const int actual_batch_size = net.batch * net.subdivisions;
    if (actual_batch_size == 1) {
        printf("\n Error: You set incorrect value batch=1 for Training! You should set batch=64 subdivision=64 \n");
        getchar();
    }else if (actual_batch_size < 8) {
        printf("\n Warning: You set batch=%d lower than 64! It is recommended to set batch=64 subdivision=64 \n", actual_batch_size);
    }

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;

    list *plist = get_paths(train_images);
    int train_images_num = plist->size;
    char **paths = (char **)list_to_array(plist);

    const int init_w = net.w;
    const int init_h = net.h;
    const int init_b = net.batch;
    int iter_save, iter_save_last, iter_map;
    iter_save = get_current_iteration(net);
    iter_save_last = get_current_iteration(net);
    iter_map = get_current_iteration(net);
    float mean_average_precision = -1;
    float best_map = mean_average_precision;

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = l.jitter;
    args.resize = l.resize;
    args.num_boxes = l.max_boxes;
    net.num_boxes = args.num_boxes;
    net.train_images_num = train_images_num;
    args.d = &buffer;
    args.type = net.data_type_;
    args.threads = 64;    // 16 or 64

    args.angle = net.angle;
    args.gaussian_noise = net.gaussian_noise;
    args.blur = net.blur;
    args.mixup = net.mixup;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.letter_box = net.letter_box;
    if (dont_show && show_imgs) show_imgs = 2;
    args.show_imgs = show_imgs;

#ifdef OPENCV
    //int num_threads = get_num_threads();
    //if(num_threads > 2) args.threads = get_num_threads() - 2;
    args.threads = 6 * ngpus;   // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge
    mat_cv* img = NULL;
    float max_img_loss = net.max_chart_loss;
    int number_of_lines = 100;
    int img_size = 1000;
    char windows_name[100];
    sprintf(windows_name, "chart_%s.png", base);
    img = draw_train_chart(windows_name, max_img_loss, net.max_batches, number_of_lines, img_size, dont_show, chart_path);
#endif    //OPENCV
    if (net.track) {
        args.track = net.track;
        args.augment_speed = net.augment_speed;
        if (net.sequential_subdivisions) args.threads = net.sequential_subdivisions * ngpus;
        else args.threads = net.subdivisions * ngpus;
        args.mini_batch = net.batch / net.time_steps;
        printf("\n Tracking! batch = %d, subdiv = %d, time_steps = %d, mini_batch = %d \n", net.batch, net.subdivisions, net.time_steps, args.mini_batch);
    }
    
    pthread_t load_thread = load_data(args);

    int count = 0;
    double time_remaining, avg_time = -1, alpha_time = 0.01;

    while (get_current_iteration(net) < net.max_batches) {
        if (l.random && count++ % 10 == 0) {
            float rand_coef = 1.4;
            if (l.random != 1.0) rand_coef = l.random;
            printf("Resizing, random_coef = %.2f \n", rand_coef);
            float random_val = rand_scale(rand_coef);    // *x or /x
            int dim_w = roundl(random_val*init_w / net.resize_step + 1) * net.resize_step;
            int dim_h = roundl(random_val*init_h / net.resize_step + 1) * net.resize_step;
            if (random_val < 1 && (dim_w > init_w || dim_h > init_h)) dim_w = init_w, dim_h = init_h;

            int max_dim_w = roundl(rand_coef*init_w / net.resize_step + 1) * net.resize_step;
            int max_dim_h = roundl(rand_coef*init_h / net.resize_step + 1) * net.resize_step;

            // at the beginning (check if enough memory) and at the end (calc rolling mean/variance)
            if (avg_loss < 0 || get_current_iteration(net) > net.max_batches - 100) {
                dim_w = max_dim_w;
                dim_h = max_dim_h;
            }

            if (dim_w < net.resize_step) dim_w = net.resize_step;
            if (dim_h < net.resize_step) dim_h = net.resize_step;
            int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
            int new_dim_b = (int)(dim_b * 0.8);
            if (new_dim_b > init_b) dim_b = new_dim_b;

            args.w = dim_w;
            args.h = dim_h;

            int k;
            if (net.dynamic_minibatch) {
                for (k = 0; k < ngpus; ++k) {
                    (*nets[k].seen) = init_b * net.subdivisions * get_current_iteration(net); // remove this line, when you will save to weights-file both: seen & cur_iteration
                    nets[k].batch = dim_b;
                    int j;
                    for (j = 0; j < nets[k].n; ++j)
                        nets[k].layers[j].batch = dim_b;
                }
                net.batch = dim_b;
                imgs = net.batch * net.subdivisions * ngpus;
                args.n = imgs;
                printf("\n %d x %d  (batch = %d) \n", dim_w, dim_h, net.batch);
            }
            else
                printf("\n %d x %d \n", dim_w, dim_h);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for (k = 0; k < ngpus; ++k) {
                resize_network(nets + k, dim_w, dim_h);
            }
            net = nets[0];
        }
        double time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        if (net.track) {
            net.sequential_subdivisions = get_current_seq_subdivisions(net);
            args.threads = net.sequential_subdivisions * ngpus;
            printf(" sequential_subdivisions = %d, sequence = %d \n", net.sequential_subdivisions, get_sequence_value(net));
        }
        load_thread = load_data(args);

        const double load_time = (what_time_is_it_now() - time);
        printf("Loaded: %lf seconds", load_time);
        if (load_time > 0.1 && avg_loss > 0) printf(" - performance bottleneck on CPU or Disk HDD/SSD");
        printf("\n");

        time = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            int wait_key = (dont_show) ? 0 : 1;
            loss = train_network_waitkey(net, train, wait_key);
        }
        else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;    // if(-inf or nan)
        avg_loss = avg_loss*.9 + loss*.1;

        const int iteration = get_current_iteration(net);
        //i = get_current_batch(net);

        int calc_map_for_each = 4 * train_images_num / (net.batch * net.subdivisions);  // calculate mAP for each 4 Epochs
        calc_map_for_each = fmax(calc_map_for_each, 100);
        int next_map_calc = iter_map + calc_map_for_each;
        next_map_calc = fmax(next_map_calc, net.burn_in);
        if (calc_map) {
            printf("\n (next mAP calculation at %d iterations) ", next_map_calc);
            if (mean_average_precision > 0) printf("\n Last accuracy mAP@0.5 = %2.2f %%, best = %2.2f %% ", mean_average_precision * 100, best_map * 100);
        }

        if (net.cudnn_half) {
            if (iteration < net.burn_in * 3) fprintf(stderr, "\n Tensor Cores are disabled until the first %d iterations are reached.\n", 3 * net.burn_in);
            else fprintf(stderr, "\n Tensor Cores are used.\n");
            fflush(stderr);
        }
        printf("\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images, %f hours left\n", iteration, loss, avg_loss, get_current_rate(net), (what_time_is_it_now() - time), iteration*imgs, avg_time);
        fflush(stdout);

        int draw_precision = 0;
        if (calc_map && (iteration >= next_map_calc || iteration == net.max_batches)) {
            if (l.random) {
                printf("Resizing to initial size: %d x %d ", init_w, init_h);
                args.w = init_w;
                args.h = init_h;
                int k;
                if (net.dynamic_minibatch) {
                    for (k = 0; k < ngpus; ++k) {
                        for (k = 0; k < ngpus; ++k) {
                            nets[k].batch = init_b;
                            int j;
                            for (j = 0; j < nets[k].n; ++j)
                                nets[k].layers[j].batch = init_b;
                        }
                    }
                    net.batch = init_b;
                    imgs = init_b * net.subdivisions * ngpus;
                    args.n = imgs;
                    printf("\n %d x %d  (batch = %d) \n", init_w, init_h, init_b);
                }
                pthread_join(load_thread, 0);
                train = buffer;
                free_data(train);
                load_thread = load_data(args);
                for (k = 0; k < ngpus; ++k) {
                    resize_network(nets + k, init_w, init_h);
                }
                net = nets[0];
            }

            copy_weights_net(net, &net_map);

            // combine Training and Validation networks
            //network net_combined = combine_train_valid_networks(net, net_map);

            iter_map = iteration;
            mean_average_precision = validate_polyGon_map(datacfg, cfgfile, weightfile, 0.25, 0.5, 0, net.letter_box, &net_map);// &net_combined);
            printf("\n mean_average_precision (mAP@0.5) = %f \n", mean_average_precision);
            if (mean_average_precision > best_map) {
                best_map = mean_average_precision;
                printf("New best mAP: %f!\n", best_map);
                char buff[256];
                sprintf(buff, "%s/%s_best.weights", backup_directory, base);
                save_weights(net, buff);
            }

            draw_precision = 1;
        }
        time_remaining = ((net.max_batches - iteration) / ngpus)*(what_time_is_it_now() - time + load_time) / 60 / 60;
        // set initial value, even if resume training from 10000 iteration
        if (avg_time < 0) avg_time = time_remaining;
        else avg_time = alpha_time * time_remaining + (1 -  alpha_time) * avg_time;
#ifdef OPENCV
        draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, net.max_batches, mean_average_precision, draw_precision, "mAP%", dont_show, mjpeg_port, avg_time);
#endif    // OPENCV

        if (iteration >= (iter_save + 10000) || iteration % 10000 == 0) {
            iter_save = iteration;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
            save_weights(net, buff);
        }

        if (iteration >= (iter_save_last + 10000) || (iteration % 10000 == 0 && iteration > 1)) {
            iter_save_last = iteration;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_last.weights", backup_directory, base);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);

#ifdef OPENCV
    release_mat(&img);
    destroy_all_windows_cv();
#endif

    // free memory
    pthread_join(load_thread, 0);
    free_data(buffer);

    free_load_threads(&args);

    free(base);
    free(paths);
    free_list_contents(plist);
    free_list(plist);

    free_list_contents_kvp(options);
    free_list(options);

    for (k = 0; k < ngpus; ++k) free_network(nets[k]);
    free(nets);
    //free_network(net);

    if (calc_map) {
        net_map.n = 0;
        free_network(net_map);
    }
}

void test_polyGon(char *datacfg, char *cfgfile, char *weightfile, 
                    char *filename, float thresh, float hier_thresh, 
                    int dont_show, int ext_output, int save_labels, 
                    char *outfile, int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if(!json_file) {
          error("fopen failed", DARKNET_LOC);
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45;
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
            image im = load_image(input, 0, 0, net.c);
            image sized;
            if(letter_box) sized = letterbox_image(im, net.w, net.h);
            else sized = resize_image(im, net.w, net.h);
            layer l = net.layers[net.n - 1];
            float *X = sized.data;
            double time = get_time_point();
            network_predict(net, X);
            printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
            int nboxes = 0;
            PolyGon_S *dets = (PolyGon_S*)xcalloc(2, sizeof(PolyGon_S));
            
            get_poly_result(net.layers[net.n - 1].output, dets, 0.5);
            
            // draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
            free(dets);
            if(outfile)
             {
                save_image(im, outfile);
             }
            else{
                save_image(im, "predictions");
            }
            if (!dont_show) {
                show_image(im, "predictions");
            }


            free_image(im);
            free_image(sized);

            if (!dont_show) {
                wait_until_press_key_cv();
                destroy_all_windows_cv();
            }

            if (filename) break;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
   
            list *plist = get_paths(input);
            char **paths = (char **)list_to_array(plist);
             printf("Start Testing!\n");
            int m = plist->size;
            //char cwd[FILEPATH_MAX];
            //getcwd(cwd,FILEPATH_MAX);
            if(access("../darknet_train/data/out",0)==-1)
            {
                if (mkdir("../darknet_train/darknet/data/out",0777))
                {
         
                    printf("creat file bag failed!!!");
                }
            }
        }
    }

    if (json_file) {
        char *tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}
#if defined(OPENCV) && defined(GPU)

// adversarial attack dnn
void draw_object_att(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);// parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    net.adversarial = 1;
    set_batch_network(&net, 1);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    //fuse_conv_batchnorm(net);
    //calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }

    srand(2222222);
    char buff[256];
    char *input = buff;

    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if (letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        image src_sized = copy_image(sized);

        layer l = net.layers[net.n - 1];
        net.num_boxes = l.max_boxes;
        int num_truth = l.truths;
        float *truth_cpu = (float *)xcalloc(num_truth, sizeof(float));

        int *it_num_set = (int *)xcalloc(1, sizeof(int));
        float *lr_set = (float *)xcalloc(1, sizeof(float));
        int *boxonly = (int *)xcalloc(1, sizeof(int));

        cv_draw_object(sized, truth_cpu, net.num_boxes, num_truth, it_num_set, lr_set, boxonly, l.classes, names);

        net.learning_rate = *lr_set;
        it_num = *it_num_set;

        float *X = sized.data;

        mat_cv* img = NULL;
        float max_img_loss = 5;
        int number_of_lines = 100;
        int img_size = 1000;
        char windows_name[100];
        char *base = basecfg(cfgfile);
        sprintf(windows_name, "chart_%s.png", base);
        img = draw_train_chart(windows_name, max_img_loss, it_num, number_of_lines, img_size, dont_show, NULL);

        int iteration;
        for (iteration = 0; iteration < it_num; ++iteration)
        {
            ocr_label_desc ocr_label;
            forward_backward_network_gpu(net, X, truth_cpu, ocr_label);

            float avg_loss = get_network_cost(net);
            draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, it_num, 0, 0, "mAP%", dont_show, 0, 0);

            float inv_loss = 1.0 / max_val_cmp(0.01, avg_loss);
            //net.learning_rate = *lr_set * inv_loss;

            if (*boxonly) {
                int dw = truth_cpu[2] * sized.w, dh = truth_cpu[3] * sized.h;
                int dx = truth_cpu[0] * sized.w - dw / 2, dy = truth_cpu[1] * sized.h - dh / 2;
                image crop = crop_image(sized, dx, dy, dw, dh);
                copy_image_inplace(src_sized, sized);
                embed_image(crop, sized, dx, dy);
            }

            show_image_cv(sized, "image_optimization");
            wait_key_cv(20);
        }

        net.train = 0;
        quantize_image(sized);
        network_predict(net, X);

        save_image_png(sized, "drawn");
        //sized = load_image("drawn.png", 0, 0, net.c);

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, 0, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(sized, dets, nboxes, thresh, names, alphabet, l.classes, 1);
        save_image(sized, "pre_predictions");
        if (!dont_show) {
            show_image(sized, "pre_predictions");
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        free_image(src_sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        free(lr_set);
        free(it_num_set);

        if (filename) break;
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}
#endif // defined(OPENCV) && defined(GPU)

void run_polyGon(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int benchmark = find_arg(argc, argv, "-benchmark");
    int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
    //if (benchmark_layers) benchmark = 1;
    if (benchmark) dont_show = 1;
    int show = find_arg(argc, argv, "-show");
    int letter_box = find_arg(argc, argv, "-letter_box");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    check_mistakes_poly = find_arg(argc, argv, "-check_mistakes");
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
    int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
    int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    char* chart_path = find_char_arg(argc, argv, "-chart", 0);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = (int)strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    if (0 == strcmp(argv[2], "test")) test_polyGon(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
    else if (0 == strcmp(argv[2], "train")) train_polyGon(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, mjpeg_port, show_imgs, benchmark_layers, chart_path);
    else if (0 == strcmp(argv[2], "map")) validate_polyGon_map(datacfg, cfg, weights, thresh, iou_thresh, map_points, letter_box, NULL);
    else if (0 == strcmp(argv[2], "draw")) {
        int it_num = 100;
        //draw_object(datacfg, cfg, weights, filename, thresh, dont_show, it_num, letter_box, benchmark_layers);
    }
    else if (0 == strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        if (filename)
            if (strlen(filename) > 0)
                if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes, avgframes, frame_skip, prefix, out_filename,
            mjpeg_port, dontdraw_bbox, json_port, dont_show, ext_output, letter_box, time_limit_sec, http_post_host, benchmark, benchmark_layers);

        free_list_contents_kvp(options);
        free_list(options);
    }
    else printf(" There isn't such command: %s", argv[2]);

    if (gpus && gpu_list && ngpus > 1) free(gpus);
}
