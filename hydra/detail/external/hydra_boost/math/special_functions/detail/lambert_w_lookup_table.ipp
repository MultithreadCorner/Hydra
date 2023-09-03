// Copyright Paul A. Bristow 2017.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// I:/modular-boost/libs/math/include/boost/math/special_functions/lambert_w_lookup_table.ipp

// A collection of 128-bit precision integral z argument Lambert W values computed using 37 decimal digits precision.
// C++ floating-point precision is 128-bit long double.
// Output as 53 decimal digits, suffixed L.

// C++ floating-point type is provided by lambert_w.hpp typedef.
// For example: typedef lookup_t double; (or float or long double)

// Written by I:\modular-boost\libs\math\test\lambert_w_lookup_table_generator.cpp Thu Jan 25 16:52:07 2018

// Sizes of arrays of z values for Lambert W[0], W[1] ... W[64]" and W[-1], W[-2] ... W[-64].

namespace hydra_boost {
namespace math {
namespace lambert_w_detail {
namespace lambert_w_lookup
{
static constexpr std::size_t  noof_sqrts = 12;
static constexpr std::size_t  noof_halves = 12;
static constexpr std::size_t  noof_w0es = 65;
static constexpr std::size_t  noof_w0zs = 65;
static constexpr std::size_t  noof_wm1es = 64;
static constexpr std::size_t  noof_wm1zs = 64;

static constexpr lookup_t halves[noof_halves] =
{ // Common to Lambert W0 and W-1 (and exactly representable).
  0.5L, 0.25L, 0.125L, 0.0625L, 0.03125L, 0.015625L, 0.0078125L, 0.00390625L, 0.001953125L, 0.0009765625L, 0.00048828125L, 0.000244140625L
}; // halves, 0.5, 0.25, ... 0.000244140625, common to W0 and W-1.

static constexpr lookup_t sqrtw0s[noof_sqrts] =
{  // For Lambert W0 only.
  0.6065306597126334242631173765403218567L, 0.77880078307140486866846070009071995L, 0.882496902584595403104717592968701829L, 0.9394130628134757862473572557999761753L, 0.9692332344763440819139583751755278177L, 0.9844964370054084060204319075254540376L, 0.9922179382602435121227899136829802692L, 0.996101369470117490071323985506950379L, 0.9980487811074754727142805899944244847L, 0.9990239141819756622368328253791383317L, 0.9995118379398893653889967919448497792L, 0.9997558891748972165136242351259789505L
}; // sqrtw0s

static constexpr lookup_t sqrtwm1s[noof_sqrts] =
{ // For Lambert W-1 only.
  1.648721270700128146848650787814163572L, 1.284025416687741484073420568062436458L, 1.133148453066826316829007227811793873L, 1.064494458917859429563390594642889673L, 1.031743407499102670938747815281507144L, 1.015747708586685747458535072082351749L, 1.007843097206447977693453559760123579L, 1.003913889338347573443609603903460282L, 1.001955033591002812046518898047477216L, 1.000977039492416535242845292611606506L, 1.000488400478694473126173623807163354L, 1.000244170429747854937005233924135774L
}; // sqrtwm1s

static constexpr lookup_t w0es[noof_w0zs] =
{ // Fukushima e powers array e[0] = 2.718, 1., e[2] = e^-1 = 0.135, e[3] = e^-2 = 0.133 ... e[64] = 4.3596100000630809736231248158884615452e-28.
  2.7182818284590452353602874713526624978e+00L,
  1.0000000000000000000000000000000000000e+00L, 3.6787944117144232159552377016146086745e-01L, 1.3533528323661269189399949497248440341e-01L, 4.9787068367863942979342415650061776632e-02L,
  1.8315638888734180293718021273241242212e-02L, 6.7379469990854670966360484231484242488e-03L, 2.4787521766663584230451674308166678915e-03L, 9.1188196555451620800313608440928262647e-04L,
  3.3546262790251183882138912578086101931e-04L, 1.2340980408667954949763669073003382607e-04L, 4.5399929762484851535591515560550610238e-05L, 1.6701700790245659312635517360580879078e-05L,
  6.1442123533282097586823081788055323112e-06L, 2.2603294069810543257852772905386894694e-06L, 8.3152871910356788406398514256526229461e-07L, 3.0590232050182578837147949770228963937e-07L,
  1.1253517471925911451377517906012719164e-07L, 4.1399377187851666596510277189552806229e-08L, 1.5229979744712628436136629233517431862e-08L, 5.6027964375372675400129828162064630798e-09L,
  2.0611536224385578279659403801558209764e-09L, 7.5825604279119067279417432412681264430e-10L, 2.7894680928689248077189130306442932077e-10L, 1.0261879631701890303927527840612497760e-10L,
  3.7751345442790977516449695475234067792e-11L, 1.3887943864964020594661763746086856910e-11L, 5.1090890280633247198744001934792157666e-12L, 1.8795288165390832947582704184221926212e-12L,
  6.9144001069402030094125846587414092712e-13L, 2.5436656473769229103033856148576816666e-13L, 9.3576229688401746049158322233787067450e-14L, 3.4424771084699764583923893328515572846e-14L,
  1.2664165549094175723120904155965096382e-14L, 4.6588861451033973641842455436101684114e-15L, 1.7139084315420129663027203425760492412e-15L, 6.3051167601469893856390211922465427614e-16L,
  2.3195228302435693883122636097380800411e-16L, 8.5330476257440657942780498229412441658e-17L, 3.1391327920480296287089646522319196491e-17L, 1.1548224173015785986262442063323868655e-17L,
  4.2483542552915889953292347828586580179e-18L, 1.5628821893349887680908829951058341550e-18L, 5.7495222642935598066643808805734234249e-19L, 2.1151310375910804866314010070226514702e-19L,
  7.7811322411337965157133167292798981918e-20L, 2.8625185805493936444701216291839372068e-20L, 1.0530617357553812378763324449428108806e-20L, 3.8739976286871871129314774972691278293e-21L,
  1.4251640827409351062853210280340602263e-21L, 5.2428856633634639371718053028323436716e-22L, 1.9287498479639177830173428165270125748e-22L, 7.0954741622847041389832693878080734877e-23L,
  2.6102790696677048047026953153318648093e-23L, 9.6026800545086760302307696700074909076e-24L, 3.5326285722008070297353928101772088374e-24L, 1.2995814250075030736007134060714855303e-24L,
  4.7808928838854690812771770423179628939e-25L, 1.7587922024243116489558751288034363178e-25L, 6.4702349256454603261540395529264893765e-26L, 2.3802664086944006058943245888024963309e-26L,
  8.7565107626965203384887328007391660366e-27L, 3.2213402859925160890012477758489437534e-27L, 1.1850648642339810062850307390972809891e-27L, 4.3596100000630809736231248158884596428e-28L,

}; // w0es

static constexpr lookup_t w0zs[noof_w0zs] =
{ // z values for W[0], W[1], W[2] ... W[64] (Fukushima array Fk).
  0.0000000000000000000000000000000000000e+00L,
  2.7182818284590452353602874713526624978e+00L, 1.4778112197861300454460854921150015626e+01L, 6.0256610769563003222785588963745153691e+01L, 2.1839260013257695631244104481144351361e+02L,
  7.4206579551288301710557790020276139812e+02L, 2.4205727609564107356503230832603296776e+03L, 7.6764321089992101948460416680168500271e+03L, 2.3847663896333826197948736795623109390e+04L,
  7.2927755348178456069389970204894839685e+04L, 2.2026465794806716516957900645284244366e+05L, 6.5861555886717600300859134371483559776e+05L, 1.9530574970280470496960624587818413980e+06L,
  5.7513740961159665432393360873381476632e+06L, 1.6836459978306874888489314790750032292e+07L, 4.9035260587081659589527825691375819733e+07L, 1.4217776832812596218820837985250320561e+08L,
  4.1063419681078006965118239806655900596e+08L, 1.1818794444719492004981570586630806042e+09L, 3.3911637183005579560532906419857313738e+09L, 9.7033039081958055593821366108308111737e+09L,
  2.7695130424147508641409976558651358487e+10L, 7.8868082614895014356985518811525255163e+10L, 2.2413047926372475980079655175092843139e+11L, 6.3573893111624333505933989166748517618e+11L,
  1.8001224834346468131040337866531539479e+12L, 5.0889698451498078710141863447784789126e+12L, 1.4365302496248562650461177217211790925e+13L, 4.0495197800161304862957327843914007993e+13L,
  1.1400869461717722015726999684446230289e+14L, 3.2059423744573386440971405952224204950e+14L, 9.0051433962267018216365614546207459567e+14L, 2.5268147258457822451512967243234631750e+15L,
  7.0832381329352301326018261305316090522e+15L, 1.9837699245933465967698692976753294646e+16L, 5.5510470830970075484537561902113104381e+16L, 1.5520433569614702817608320254284931407e+17L,
  4.3360826779369661842459877227403719730e+17L, 1.2105254067703227363724895246493485480e+18L, 3.3771426165357561311906703760513324357e+18L, 9.4154106734807994163159964299613921804e+18L,
  2.6233583234732252918129199544138403574e+19L, 7.3049547543861043990576614751671879498e+19L, 2.0329709713386190214340167519800405595e+20L, 5.6547040503180956413560918381429636734e+20L,
  1.5720421975868292906615658755032744790e+21L, 4.3682149334771264822761478593874428627e+21L, 1.2132170565093316762294432610117848880e+22L, 3.3680332378068632345542636794533635462e+22L,
  9.3459982052259884835729892206738573922e+22L, 2.5923527642935362320437266614667426924e+23L, 7.1876803203773878618909930893087860822e+23L, 1.9921241603726199616378561653688236827e+24L,
  5.5192924995054165325072406547517121131e+24L, 1.5286067837683347062387143159276002521e+25L, 4.2321318958281094260005100745711666956e+25L, 1.1713293177672778461879598480402173158e+26L,
  3.2408603996214813669049988277609543829e+26L, 8.9641258264226027960478448084812796397e+26L, 2.4787141382364034104243901241243054434e+27L, 6.8520443388941057019777430988685937812e+27L,
  1.8936217407781711443114787060753312270e+28L, 5.2317811346197017832254642778313331353e+28L, 1.4450833904658542238325922893692265683e+29L, 3.9904954117194348050619127737142206367e+29L

}; // w0zs

static constexpr lookup_t wm1es[noof_wm1es] =
{ // Fukushima e array e[0] = e^1 = 2.718, e[1] = e^2 = 7.39 ... e[64] = 4.60718e+28.
  2.7182818284590452353602874713526624978e+00L,
  7.3890560989306502272304274605750078132e+00L, 2.0085536923187667740928529654581717897e+01L, 5.4598150033144239078110261202860878403e+01L, 1.4841315910257660342111558004055227962e+02L,
  4.0342879349273512260838718054338827961e+02L, 1.0966331584284585992637202382881214324e+03L, 2.9809579870417282747435920994528886738e+03L, 8.1030839275753840077099966894327599650e+03L,
  2.2026465794806716516957900645284244366e+04L, 5.9874141715197818455326485792257781614e+04L, 1.6275479141900392080800520489848678317e+05L, 4.4241339200892050332610277594908828178e+05L,
  1.2026042841647767777492367707678594494e+06L, 3.2690173724721106393018550460917213155e+06L, 8.8861105205078726367630237407814503508e+06L, 2.4154952753575298214775435180385823880e+07L,
  6.5659969137330511138786503259060033569e+07L, 1.7848230096318726084491003378872270388e+08L, 4.8516519540979027796910683054154055868e+08L, 1.3188157344832146972099988837453027851e+09L,
  3.5849128461315915616811599459784206892e+09L, 9.7448034462489026000346326848229752776e+09L, 2.6489122129843472294139162152811882341e+10L, 7.2004899337385872524161351466126157915e+10L,
  1.9572960942883876426977639787609534279e+11L, 5.3204824060179861668374730434117744166e+11L, 1.4462570642914751736770474229969288569e+12L, 3.9313342971440420743886205808435276858e+12L,
  1.0686474581524462146990468650741401650e+13L, 2.9048849665247425231085682111679825667e+13L, 7.8962960182680695160978022635108224220e+13L, 2.1464357978591606462429776153126088037e+14L,
  5.8346174252745488140290273461039101900e+14L, 1.5860134523134307281296446257746601252e+15L, 4.3112315471151952271134222928569253908e+15L, 1.1719142372802611308772939791190194522e+16L,
  3.1855931757113756220328671701298646000e+16L, 8.6593400423993746953606932719264934250e+16L, 2.3538526683701998540789991074903480451e+17L, 6.3984349353005494922266340351557081888e+17L,
  1.7392749415205010473946813036112352261e+18L, 4.7278394682293465614744575627442803708e+18L, 1.2851600114359308275809299632143099258e+19L, 3.4934271057485095348034797233406099533e+19L,
  9.4961194206024488745133649117118323102e+19L, 2.5813128861900673962328580021527338043e+20L, 7.0167359120976317386547159988611740546e+20L, 1.9073465724950996905250998409538484474e+21L,
  5.1847055285870724640874533229334853848e+21L, 1.4093490824269387964492143312370168789e+22L, 3.8310080007165768493035695487861993899e+22L, 1.0413759433029087797183472933493796440e+23L,
  2.8307533032746939004420635480140745409e+23L, 7.6947852651420171381827455901293939921e+23L, 2.0916594960129961539070711572146737782e+24L, 5.6857199993359322226403488206332533034e+24L,
  1.5455389355901039303530766911174620068e+25L, 4.2012104037905142549565934307191617684e+25L, 1.1420073898156842836629571831447656302e+26L, 3.1042979357019199087073421411071003721e+26L,
  8.4383566687414544890733294803731179601e+26L, 2.2937831594696098790993528402686136005e+27L, 6.2351490808116168829092387089284697448e+27L
}; // wm1es

static constexpr lookup_t wm1zs[noof_wm1zs] =
{ // Fukushima G array of z values for integral K, (Fukushima Gk) g[0] (k = -1) = 1 ... g[64] = -1.0264389699511303e-26.
  -3.6787944117144232159552377016146086745e-01L,
  -2.7067056647322538378799898994496880682e-01L, -1.4936120510359182893802724695018532990e-01L, -7.3262555554936721174872085092964968848e-02L, -3.3689734995427335483180242115742121244e-02L,
  -1.4872513059998150538271004584900007349e-02L, -6.3831737588816134560219525908649783853e-03L, -2.6837010232200947105711130062468881545e-03L, -1.1106882367801159454787302165703044346e-03L,
  -4.5399929762484851535591515560550610238e-04L, -1.8371870869270225243899069096638966986e-04L, -7.3730548239938517104187698145666387735e-05L, -2.9384282290753706235208604777002963102e-05L,
  -1.1641402067449950376895791995913672125e-05L, -4.5885348075273868255721924655343445906e-06L, -1.8005627955081458322204028649620350662e-06L, -7.0378941219347833214067471222239770590e-07L,
  -2.7413963540482731185045932620331377352e-07L, -1.0645313231320808326024667350792279852e-07L, -4.1223072448771156559318807603116419528e-08L, -1.5923376898615004128677660806663065530e-08L,
  -6.1368298043116345769816086674174450569e-09L, -2.3602323152914347699033314033408744848e-09L, -9.0603229062698346039479269140561762700e-10L, -3.4719859662410051486654409365217142276e-10L,
  -1.3283631472964644271673440503045960993e-10L, -5.0747278046555248958473301297399200773e-11L, -1.9360320299432568426355237044475945959e-11L, -7.3766303773930764398798182830872768331e-12L,
  -2.8072868906520523814747496670136120235e-12L, -1.0671679036256927021016406931839827582e-12L, -4.0525329757101362313986893299088308423e-13L, -1.5374324278841211301808010293913555758e-13L,
  -5.8272886672428440854292491647585674200e-14L, -2.2067908660514462849736574172862899665e-14L, -8.3502821888768497979241489950570881481e-15L, -3.1572276215253043438828784344882603413e-15L,
  -1.1928704609782512589094065678481294667e-15L, -4.5038074274761565346423524046963087756e-16L, -1.6993417021166355981316939131434632072e-16L, -6.4078169762734539491726202799339200356e-17L,
  -2.4147993510032951187990399698408378385e-17L, -9.0950634616416460925150243301974013218e-18L, -3.4236981860988704669138593608831552044e-18L, -1.2881333612472271400115547331327717431e-18L,
  -4.8440839844747536942311292467369300505e-19L, -1.8207788854829779430777944237164900798e-19L, -6.8407875971564885101695409345634890862e-20L, -2.5690139750480973292141845983878483991e-20L,
  -9.6437492398195889150867140826350628738e-21L, -3.6186918227651991108814673877821174787e-21L, -1.3573451162272064984454015639725697008e-21L, -5.0894204288895982960223079251039701810e-22L,
  -1.9076194289884357960571121174956927722e-22L, -7.1476978375412669048039237333931704166e-23L, -2.6773000149758626855152191436980592206e-23L, -1.0025115553818576399048488234179587012e-23L,
  -3.7527362568743669891693429406973638384e-24L, -1.4043571811296963574776515073934728352e-24L, -5.2539064576179122030932396804434996219e-25L, -1.9650175744554348142907611432678556896e-25L,
  -7.3474021582506822389671905824031421322e-26L, -2.7465543000397410133825686340097295749e-26L, -1.0264389699511282259046957018510946438e-26L
}; // wm1zs
} // namespace lambert_w_lookup
} // namespace detail
} // namespace math
} // namespace hydra_boost