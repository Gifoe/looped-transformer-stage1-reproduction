# K2 to K4 propagation

{
  "development": {
    "K2_rescues": 2,
    "K2_rescues_correct_at_K4": 0,
    "new_correct_all_native_loops_wrong": 0
  },
  "confirmation": {
    "K2_rescues": 5,
    "K2_rescues_correct_at_K4": 4,
    "new_correct_all_native_loops_wrong": 1
  }
}

Full candidate distributions and per-question persistence are in PROPAGATION.json. K2 rescues correct at K4 may already have a correct native K4; final rescue and strict final rescue are recorded separately.

L10-16 readout attention increments are the actual input_layernorm_2 outputs before residual addition, observed without overwriting. ATTENTION_ASSOCIATIONS.json compares their induced vector changes with same-sign actual-dose random controls, in all questions and separately gold-defined rescue/damage groups. These are descriptive associations, not causal mediation or a label-free predictor. Original per-question BF16 observation tensors remain on the server under hashes.

{
  "development:all": {
    "n": 256,
    "layers": {
      "2:10": {
        "selected_mean_delta_L2": 0.05637819410361026,
        "random_mean_delta_L2": 0.06376304717092526,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.007384853067315003,
          "descriptive_ci95": [
            -0.008072564818728539,
            -0.006676905215567356
          ],
          "bonferroni21_two_sided_ci": [
            -0.008454487562873203,
            -0.006364002215435853
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:11": {
        "selected_mean_delta_L2": 0.09165691872266099,
        "random_mean_delta_L2": 0.09835661289629097,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.006699694173629997,
          "descriptive_ci95": [
            -0.00832878802401652,
            -0.0050184013902857295
          ],
          "bonferroni21_two_sided_ci": [
            -0.009207366400363693,
            -0.004181378713790248
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:12": {
        "selected_mean_delta_L2": 0.10224929276845765,
        "random_mean_delta_L2": 0.10815005149010422,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.00590075872164657,
          "descriptive_ci95": [
            -0.007543838719311067,
            -0.004236895879618147
          ],
          "bonferroni21_two_sided_ci": [
            -0.008461330907442806,
            -0.0034549892579253567
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:13": {
        "selected_mean_delta_L2": 0.08517630674074335,
        "random_mean_delta_L2": 0.08912389101076938,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.003947584270026043,
          "descriptive_ci95": [
            -0.005425471700712671,
            -0.0025369347010541616
          ],
          "bonferroni21_two_sided_ci": [
            -0.006264149214108881,
            -0.001843504983643568
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:14": {
        "selected_mean_delta_L2": 0.10963028197207694,
        "random_mean_delta_L2": 0.11214085723417164,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.0025105752620946937,
          "descriptive_ci95": [
            -0.004355270242055804,
            -0.0006718423861654656
          ],
          "bonferroni21_two_sided_ci": [
            -0.005414775946562221,
            0.0001719696135580457
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:15": {
        "selected_mean_delta_L2": 0.10372112906245658,
        "random_mean_delta_L2": 0.10205430261754356,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.0016668264449130061,
          "descriptive_ci95": [
            -0.00017733469135443086,
            0.0035940511117606233
          ],
          "bonferroni21_two_sided_ci": [
            -0.0013686448939186906,
            0.004525575526106735
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:16": {
        "selected_mean_delta_L2": 0.053565551600928955,
        "random_mean_delta_L2": 0.05137198488745659,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.0021935667134723633,
          "descriptive_ci95": [
            0.0009643853902584971,
            0.003508964572412156
          ],
          "bonferroni21_two_sided_ci": [
            0.00032713311525863067,
            0.004461663023435428
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:10": {
        "selected_mean_delta_L2": 0.061062533186492296,
        "random_mean_delta_L2": 0.048486598269820186,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.012575934916672107,
          "descriptive_ci95": [
            0.010466020834267233,
            0.014847250779368767
          ],
          "bonferroni21_two_sided_ci": [
            0.009483378568028649,
            0.01606159841303032
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:11": {
        "selected_mean_delta_L2": 0.09404982957810264,
        "random_mean_delta_L2": 0.07637261079741081,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.017677218780691825,
          "descriptive_ci95": [
            0.014566840662128107,
            0.020954982051844314
          ],
          "bonferroni21_two_sided_ci": [
            0.012993772334592427,
            0.022728341626767942
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:12": {
        "selected_mean_delta_L2": 0.09795101503690729,
        "random_mean_delta_L2": 0.07707886489840278,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.020872150138504506,
          "descriptive_ci95": [
            0.017340391579584957,
            0.024581613777385082
          ],
          "bonferroni21_two_sided_ci": [
            0.015841074263068654,
            0.02649294710154637
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:13": {
        "selected_mean_delta_L2": 0.07239175614228448,
        "random_mean_delta_L2": 0.054756195485967385,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.017635560656317096,
          "descriptive_ci95": [
            0.014912706382695779,
            0.020513442163482477
          ],
          "bonferroni21_two_sided_ci": [
            0.01393736832043985,
            0.022059208447846024
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:14": {
        "selected_mean_delta_L2": 0.09538890559061425,
        "random_mean_delta_L2": 0.07635786277556272,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.01903104281505153,
          "descriptive_ci95": [
            0.015407220800734428,
            0.022680591003151136
          ],
          "bonferroni21_two_sided_ci": [
            0.013554733001431975,
            0.024416325589350512
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:15": {
        "selected_mean_delta_L2": 0.08612992098680422,
        "random_mean_delta_L2": 0.06791238966217586,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.018217531324628382,
          "descriptive_ci95": [
            0.014982675637927935,
            0.021846689965728607
          ],
          "bonferroni21_two_sided_ci": [
            0.013408903478486247,
            0.023772734878223154
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:16": {
        "selected_mean_delta_L2": 0.038216431037743206,
        "random_mean_delta_L2": 0.030871154078573642,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.007345276959169561,
          "descriptive_ci95": [
            0.005766116898142419,
            0.009111029958969749
          ],
          "bonferroni21_two_sided_ci": [
            0.004983030911108837,
            0.009806060473250743
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:10": {
        "selected_mean_delta_L2": 0.056606990216298676,
        "random_mean_delta_L2": 0.0471154781030901,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.009491512113208575,
          "descriptive_ci95": [
            0.006916082485051085,
            0.012098799861414362
          ],
          "bonferroni21_two_sided_ci": [
            0.005218645362628175,
            0.01369387752536697
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:11": {
        "selected_mean_delta_L2": 0.08228169255857296,
        "random_mean_delta_L2": 0.06707933692464138,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.015202355633931568,
          "descriptive_ci95": [
            0.011273097858094591,
            0.019234827990506395
          ],
          "bonferroni21_two_sided_ci": [
            0.009307887246493366,
            0.021133460301070557
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:12": {
        "selected_mean_delta_L2": 0.077081107846421,
        "random_mean_delta_L2": 0.0627484547988766,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.0143326530475444,
          "descriptive_ci95": [
            0.010329679809311758,
            0.01835275191868544
          ],
          "bonferroni21_two_sided_ci": [
            0.0077276147630589535,
            0.020793515453532176
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:13": {
        "selected_mean_delta_L2": 0.05771481806178471,
        "random_mean_delta_L2": 0.0476226187471126,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.010092199314672114,
          "descriptive_ci95": [
            0.006520193580544483,
            0.0135505372875621
          ],
          "bonferroni21_two_sided_ci": [
            0.004247908487159317,
            0.015215646929554794
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:14": {
        "selected_mean_delta_L2": 0.06835175457545985,
        "random_mean_delta_L2": 0.05670946421444768,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.01164229036101217,
          "descriptive_ci95": [
            0.007862063530471345,
            0.015303942070755087
          ],
          "bonferroni21_two_sided_ci": [
            0.0058375478073604365,
            0.0176080481798159
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:15": {
        "selected_mean_delta_L2": 0.06179692900317105,
        "random_mean_delta_L2": 0.05327268792762596,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.008524241075545084,
          "descriptive_ci95": [
            0.005596399162517384,
            0.011424876046563953
          ],
          "bonferroni21_two_sided_ci": [
            0.003942189108810998,
            0.013017833350540709
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:16": {
        "selected_mean_delta_L2": 0.028966594238598917,
        "random_mean_delta_L2": 0.024700965065976584,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.004265629172622336,
          "descriptive_ci95": [
            0.002745293367006829,
            0.0058026605719963585
          ],
          "bonferroni21_two_sided_ci": [
            0.0019137136246360359,
            0.006705641831930031
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      }
    }
  },
  "development:strict_K4_rescue": {
    "n": 0,
    "layers": {}
  },
  "development:damage": {
    "n": 0,
    "layers": {}
  },
  "development:other": {
    "n": 256,
    "layers": {
      "2:10": {
        "selected_mean_delta_L2": 0.05637819410361026,
        "random_mean_delta_L2": 0.06376304717092526
      },
      "2:11": {
        "selected_mean_delta_L2": 0.09165691872266099,
        "random_mean_delta_L2": 0.09835661289629097
      },
      "2:12": {
        "selected_mean_delta_L2": 0.10224929276845765,
        "random_mean_delta_L2": 0.10815005149010422
      },
      "2:13": {
        "selected_mean_delta_L2": 0.08517630674074335,
        "random_mean_delta_L2": 0.08912389101076938
      },
      "2:14": {
        "selected_mean_delta_L2": 0.10963028197207694,
        "random_mean_delta_L2": 0.11214085723417164
      },
      "2:15": {
        "selected_mean_delta_L2": 0.10372112906245658,
        "random_mean_delta_L2": 0.10205430261754356
      },
      "2:16": {
        "selected_mean_delta_L2": 0.053565551600928955,
        "random_mean_delta_L2": 0.05137198488745659
      },
      "3:10": {
        "selected_mean_delta_L2": 0.061062533186492296,
        "random_mean_delta_L2": 0.048486598269820186
      },
      "3:11": {
        "selected_mean_delta_L2": 0.09404982957810264,
        "random_mean_delta_L2": 0.07637261079741081
      },
      "3:12": {
        "selected_mean_delta_L2": 0.09795101503690729,
        "random_mean_delta_L2": 0.07707886489840278
      },
      "3:13": {
        "selected_mean_delta_L2": 0.07239175614228448,
        "random_mean_delta_L2": 0.054756195485967385
      },
      "3:14": {
        "selected_mean_delta_L2": 0.09538890559061425,
        "random_mean_delta_L2": 0.07635786277556272
      },
      "3:15": {
        "selected_mean_delta_L2": 0.08612992098680422,
        "random_mean_delta_L2": 0.06791238966217586
      },
      "3:16": {
        "selected_mean_delta_L2": 0.038216431037743206,
        "random_mean_delta_L2": 0.030871154078573642
      },
      "4:10": {
        "selected_mean_delta_L2": 0.056606990216298676,
        "random_mean_delta_L2": 0.0471154781030901
      },
      "4:11": {
        "selected_mean_delta_L2": 0.08228169255857296,
        "random_mean_delta_L2": 0.06707933692464138
      },
      "4:12": {
        "selected_mean_delta_L2": 0.077081107846421,
        "random_mean_delta_L2": 0.0627484547988766
      },
      "4:13": {
        "selected_mean_delta_L2": 0.05771481806178471,
        "random_mean_delta_L2": 0.0476226187471126
      },
      "4:14": {
        "selected_mean_delta_L2": 0.06835175457545985,
        "random_mean_delta_L2": 0.05670946421444768
      },
      "4:15": {
        "selected_mean_delta_L2": 0.06179692900317105,
        "random_mean_delta_L2": 0.05327268792762596
      },
      "4:16": {
        "selected_mean_delta_L2": 0.028966594238598917,
        "random_mean_delta_L2": 0.024700965065976584
      }
    }
  },
  "confirmation:all": {
    "n": 512,
    "layers": {
      "2:10": {
        "selected_mean_delta_L2": 0.05651311466299654,
        "random_mean_delta_L2": 0.06375802191740904,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.007244907254412486,
          "descriptive_ci95": [
            -0.007788459636768271,
            -0.006702221188764046
          ],
          "bonferroni21_two_sided_ci": [
            -0.008049665556777795,
            -0.006420828989608166
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:11": {
        "selected_mean_delta_L2": 0.0902519550966347,
        "random_mean_delta_L2": 0.09961812424072179,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.009366169144087092,
          "descriptive_ci95": [
            -0.010448864480359796,
            -0.008254168386790058
          ],
          "bonferroni21_two_sided_ci": [
            -0.011098623265152764,
            -0.007664839729974592
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:12": {
        "selected_mean_delta_L2": 0.10247418714076967,
        "random_mean_delta_L2": 0.1095117377726062,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.0070375506318365415,
          "descriptive_ci95": [
            -0.008256106560709228,
            -0.0058354524398892315
          ],
          "bonferroni21_two_sided_ci": [
            -0.008900681604233705,
            -0.005080692012106368
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:13": {
        "selected_mean_delta_L2": 0.08637010273235449,
        "random_mean_delta_L2": 0.09205865866361389,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.005688555931259397,
          "descriptive_ci95": [
            -0.006739022587008304,
            -0.004582702618272302
          ],
          "bonferroni21_two_sided_ci": [
            -0.007253380959217077,
            -0.004010274990697373
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:14": {
        "selected_mean_delta_L2": 0.11280412781497604,
        "random_mean_delta_L2": 0.11560142653717964,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.0027972987222036018,
          "descriptive_ci95": [
            -0.004290872096863606,
            -0.001356722753640937
          ],
          "bonferroni21_two_sided_ci": [
            -0.005018740962492688,
            -0.0004961791015536347
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:15": {
        "selected_mean_delta_L2": 0.10389451194033184,
        "random_mean_delta_L2": 0.10411983332625432,
        "paired_selected_minus_random_delta_norm": {
          "mean": -0.0002253213859224943,
          "descriptive_ci95": [
            -0.0015632286074140072,
            0.0011111063689000238
          ],
          "bonferroni21_two_sided_ci": [
            -0.002193779113219587,
            0.0017403398153991556
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "2:16": {
        "selected_mean_delta_L2": 0.05596494065380406,
        "random_mean_delta_L2": 0.054328417459084954,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.0016365231947191006,
          "descriptive_ci95": [
            0.0006637906056682192,
            0.002649929175146138
          ],
          "bonferroni21_two_sided_ci": [
            0.00012250000377029242,
            0.0032129631677211123
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:10": {
        "selected_mean_delta_L2": 0.0616696343322377,
        "random_mean_delta_L2": 0.049719871858260135,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.011949762473977566,
          "descriptive_ci95": [
            0.010673866201016173,
            0.01324154232626544
          ],
          "bonferroni21_two_sided_ci": [
            0.010111574889758064,
            0.014062594463128124
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:11": {
        "selected_mean_delta_L2": 0.09562993195327812,
        "random_mean_delta_L2": 0.07655733696544993,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.019072594987828196,
          "descriptive_ci95": [
            0.016896627755482993,
            0.0211316649560531
          ],
          "bonferroni21_two_sided_ci": [
            0.01597538701948709,
            0.022370780600358603
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:12": {
        "selected_mean_delta_L2": 0.09916681306538339,
        "random_mean_delta_L2": 0.07906180363517722,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.020105009430206194,
          "descriptive_ci95": [
            0.01787391870997589,
            0.022348378468443664
          ],
          "bonferroni21_two_sided_ci": [
            0.01667336107608666,
            0.02361179307262508
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:13": {
        "selected_mean_delta_L2": 0.07370093549829956,
        "random_mean_delta_L2": 0.05778622006744249,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.01591471543085707,
          "descriptive_ci95": [
            0.013781462013097878,
            0.01837850015965023
          ],
          "bonferroni21_two_sided_ci": [
            0.01257302479899644,
            0.019696526217129452
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:14": {
        "selected_mean_delta_L2": 0.09720978297715471,
        "random_mean_delta_L2": 0.07922781866348068,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.01798196431367403,
          "descriptive_ci95": [
            0.015288562550318744,
            0.02071549005135839
          ],
          "bonferroni21_two_sided_ci": [
            0.013808607191478589,
            0.02223896707052899
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:15": {
        "selected_mean_delta_L2": 0.0845583722788126,
        "random_mean_delta_L2": 0.06993454938543636,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.014623822893376225,
          "descriptive_ci95": [
            0.01247828652642589,
            0.01690642632920659
          ],
          "bonferroni21_two_sided_ci": [
            0.011154654928523959,
            0.01816979736655651
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "3:16": {
        "selected_mean_delta_L2": 0.0383875474178174,
        "random_mean_delta_L2": 0.03156699182841308,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.006820555589404317,
          "descriptive_ci95": [
            0.005754953402224491,
            0.007849974723009965
          ],
          "bonferroni21_two_sided_ci": [
            0.005282285383440511,
            0.008484022076019216
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:10": {
        "selected_mean_delta_L2": 0.055842300584138055,
        "random_mean_delta_L2": 0.04860100668503184,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.007241293899106213,
          "descriptive_ci95": [
            0.0044844415576157535,
            0.009942745479929349
          ],
          "bonferroni21_two_sided_ci": [
            0.0028383312455498117,
            0.011241372714528554
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:11": {
        "selected_mean_delta_L2": 0.08258151538021735,
        "random_mean_delta_L2": 0.06918473510201043,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.013396780278206913,
          "descriptive_ci95": [
            0.010205097829396598,
            0.016744775422919734
          ],
          "bonferroni21_two_sided_ci": [
            0.00840999636043427,
            0.018929617034244905
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:12": {
        "selected_mean_delta_L2": 0.07631832683162544,
        "random_mean_delta_L2": 0.06390302443315041,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.012415302398475037,
          "descriptive_ci95": [
            0.008944866845627888,
            0.015949796265798805
          ],
          "bonferroni21_two_sided_ci": [
            0.0075478548236497,
            0.017658234636635736
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:13": {
        "selected_mean_delta_L2": 0.05818376679089028,
        "random_mean_delta_L2": 0.05008380837162708,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.008099958419263208,
          "descriptive_ci95": [
            0.005112272714498991,
            0.01101846207654074
          ],
          "bonferroni21_two_sided_ci": [
            0.0031056376036387064,
            0.012630230846352947
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:14": {
        "selected_mean_delta_L2": 0.0673258307745988,
        "random_mean_delta_L2": 0.05975788396859447,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.007567946806004323,
          "descriptive_ci95": [
            0.0037110214595388615,
            0.011192473696636479
          ],
          "bonferroni21_two_sided_ci": [
            0.0010849623430872707,
            0.013384248651896346
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:15": {
        "selected_mean_delta_L2": 0.06181517426880424,
        "random_mean_delta_L2": 0.05564640881809172,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.006168765450712516,
          "descriptive_ci95": [
            0.0034937146436167807,
            0.008811568862209714
          ],
          "bonferroni21_two_sided_ci": [
            0.002088777896599433,
            0.010112651160217612
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      },
      "4:16": {
        "selected_mean_delta_L2": 0.02799068450339233,
        "random_mean_delta_L2": 0.024866795875169385,
        "paired_selected_minus_random_delta_norm": {
          "mean": 0.003123888628222945,
          "descriptive_ci95": [
            0.0019976063160586912,
            0.0043128213924721525
          ],
          "bonferroni21_two_sided_ci": [
            0.0013766348021491577,
            0.004901463125634934
          ],
          "scope": "21 observational layer/depth contrasts per cohort; not causal or an accuracy gate"
        }
      }
    }
  },
  "confirmation:strict_K4_rescue": {
    "n": 0,
    "layers": {}
  },
  "confirmation:damage": {
    "n": 0,
    "layers": {}
  },
  "confirmation:other": {
    "n": 512,
    "layers": {
      "2:10": {
        "selected_mean_delta_L2": 0.05651311466299654,
        "random_mean_delta_L2": 0.06375802191740904
      },
      "2:11": {
        "selected_mean_delta_L2": 0.0902519550966347,
        "random_mean_delta_L2": 0.09961812424072179
      },
      "2:12": {
        "selected_mean_delta_L2": 0.10247418714076967,
        "random_mean_delta_L2": 0.1095117377726062
      },
      "2:13": {
        "selected_mean_delta_L2": 0.08637010273235449,
        "random_mean_delta_L2": 0.09205865866361389
      },
      "2:14": {
        "selected_mean_delta_L2": 0.11280412781497604,
        "random_mean_delta_L2": 0.11560142653717964
      },
      "2:15": {
        "selected_mean_delta_L2": 0.10389451194033184,
        "random_mean_delta_L2": 0.10411983332625432
      },
      "2:16": {
        "selected_mean_delta_L2": 0.05596494065380406,
        "random_mean_delta_L2": 0.054328417459084954
      },
      "3:10": {
        "selected_mean_delta_L2": 0.0616696343322377,
        "random_mean_delta_L2": 0.049719871858260135
      },
      "3:11": {
        "selected_mean_delta_L2": 0.09562993195327812,
        "random_mean_delta_L2": 0.07655733696544993
      },
      "3:12": {
        "selected_mean_delta_L2": 0.09916681306538339,
        "random_mean_delta_L2": 0.07906180363517722
      },
      "3:13": {
        "selected_mean_delta_L2": 0.07370093549829956,
        "random_mean_delta_L2": 0.05778622006744249
      },
      "3:14": {
        "selected_mean_delta_L2": 0.09720978297715471,
        "random_mean_delta_L2": 0.07922781866348068
      },
      "3:15": {
        "selected_mean_delta_L2": 0.0845583722788126,
        "random_mean_delta_L2": 0.06993454938543636
      },
      "3:16": {
        "selected_mean_delta_L2": 0.0383875474178174,
        "random_mean_delta_L2": 0.03156699182841308
      },
      "4:10": {
        "selected_mean_delta_L2": 0.055842300584138055,
        "random_mean_delta_L2": 0.04860100668503184
      },
      "4:11": {
        "selected_mean_delta_L2": 0.08258151538021735,
        "random_mean_delta_L2": 0.06918473510201043
      },
      "4:12": {
        "selected_mean_delta_L2": 0.07631832683162544,
        "random_mean_delta_L2": 0.06390302443315041
      },
      "4:13": {
        "selected_mean_delta_L2": 0.05818376679089028,
        "random_mean_delta_L2": 0.05008380837162708
      },
      "4:14": {
        "selected_mean_delta_L2": 0.0673258307745988,
        "random_mean_delta_L2": 0.05975788396859447
      },
      "4:15": {
        "selected_mean_delta_L2": 0.06181517426880424,
        "random_mean_delta_L2": 0.05564640881809172
      },
      "4:16": {
        "selected_mean_delta_L2": 0.02799068450339233,
        "random_mean_delta_L2": 0.024866795875169385
      }
    }
  }
}

## Strict early rescues

{
  "development": {
    "strict_K2_rescues": 0,
    "still_correct_at_K4": 0,
    "new_strict_K4_rescues": 0,
    "rows": []
  },
  "confirmation": {
    "strict_K2_rescues": 3,
    "still_correct_at_K4": 3,
    "new_strict_K4_rescues": 0,
    "rows": [
      {
        "id": "Mercury_7213360",
        "native_K4_correct": true,
        "modified_K3_correct": true,
        "modified_K4_correct": true,
        "becomes_new_strict_K4_rescue": false
      },
      {
        "id": "Mercury_SC_413531",
        "native_K4_correct": true,
        "modified_K3_correct": true,
        "modified_K4_correct": true,
        "becomes_new_strict_K4_rescue": false
      },
      {
        "id": "TIMSS_2011_4_pg67",
        "native_K4_correct": true,
        "modified_K3_correct": true,
        "modified_K4_correct": true,
        "becomes_new_strict_K4_rescue": false
      }
    ]
  }
}

A still-correct modified K4 is not a new K4 rescue when native K4 was already correct. No selected Q correction produces a strict K4 rescue in either cohort.
