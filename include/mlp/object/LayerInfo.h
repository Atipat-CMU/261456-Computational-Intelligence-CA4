#ifndef LAYERINFO_H
#define LAYERINFO_H

namespace mlp {
    enum ly_type {
        INPUT,
        HIDDEN,
        OUTPUT
    };

    struct layer_info {
        ly_type type = HIDDEN;
        double (*activation)(double, bool);
        int N_node;
    };
}

#endif
