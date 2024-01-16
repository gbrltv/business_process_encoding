def enc_selector(encoding):
    if encoding == "onehot":
        from encode.baseline import run_onehot as run

        return run
    elif encoding == "count2vec":
        from encode.baseline import run_count2vec as run

        return run
    elif encoding == "alignment":
        from encode.pm import run_alignment as run

        return run
    elif encoding == "logskeleton":
        from encode.pm import run_logskeleton as run

        return run
    elif encoding == "tokenreplay":
        from encode.pm import run_tokenreplay as run

        return run
    elif encoding == "tfidf":
        from encode.text import run_tfidf as run

        return run
    elif encoding == "hash2vec":
        from encode.text import run_hash2vec as run

        return run
    elif encoding == "doc2vec":
        from encode.text import run_doc2vec as run

        return run
    elif encoding == "word2vec":
        from encode.text import run_word2vec as run

        return run
    elif encoding == "boostne":
        from encode.graph import run_boostne as run

        return run
    elif encoding == "deepwalk":
        from encode.graph import run_deepwalk as run

        return run
    elif encoding == "diff2vec":
        from encode.graph import run_diff2vec as run

        return run
    elif encoding == "glee":
        from encode.graph import run_glee as run

        return run
    elif encoding == "grarep":
        from encode.graph import run_grarep as run

        return run
    elif encoding == "hope":
        from encode.graph import run_hope as run

        return run
    elif encoding == "laplacianeigenmaps":
        from encode.graph import run_laplacianeigenmaps as run

        return run
    elif encoding == "netmf":
        from encode.graph import run_netmf as run

        return run
    elif encoding == "nmfadmm":
        from encode.graph import run_nmfadmm as run

        return run
    elif encoding == "node2vec":
        from encode.graph import run_node2vec as run

        return run
    elif encoding == "nodesketch":
        from encode.graph import run_nodesketch as run

        return run
    elif encoding == "role2vec":
        from encode.graph import run_role2vec as run

        return run
    elif encoding == "walklets":
        from encode.graph import run_walklets as run

        return run
    else:
        raise Exception(
            "Please select an existing encoding among:\nonehot\ncount2vec\nalignment\nlogskeleton\ntokenreplay\ntfidf\nhash2vec\ndoc2vec\nword2vec\nboostne\ndeepwalk\ndiff2vec\nglee\ngrarep\nhope\nlaplacianeigenmaps\nnetmf\nnmfadmm\nnode2vec\nnodesketch\nrole2vec\nwalklets"
        )
