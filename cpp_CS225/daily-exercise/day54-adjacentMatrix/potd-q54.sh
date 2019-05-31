#!/bin/sh
# This script was generated using Makeself 2.3.0

ORIG_UMASK=`umask`
if test "n" = n; then
    umask 077
fi

CRCsum="1211792870"
MD5="59b4f7e08ea12f51ba829f1cc8d25c0c"
TMPROOT=${TMPDIR:=/tmp}
USER_PWD="$PWD"; export USER_PWD

label="Extracting potd-q54"
script="echo"
scriptargs="The initial files can be found in the newly created directory: potd-q54"
licensetxt=""
helpheader=''
targetdir="potd-q54"
filesizes="6031"
keep="y"
nooverwrite="n"
quiet="n"
nodiskspace="n"

print_cmd_arg=""
if type printf > /dev/null; then
    print_cmd="printf"
elif test -x /usr/ucb/echo; then
    print_cmd="/usr/ucb/echo"
else
    print_cmd="echo"
fi

unset CDPATH

MS_Printf()
{
    $print_cmd $print_cmd_arg "$1"
}

MS_PrintLicense()
{
  if test x"$licensetxt" != x; then
    echo "$licensetxt"
    while true
    do
      MS_Printf "Please type y to accept, n otherwise: "
      read yn
      if test x"$yn" = xn; then
        keep=n
	eval $finish; exit 1
        break;
      elif test x"$yn" = xy; then
        break;
      fi
    done
  fi
}

MS_diskspace()
{
	(
	if test -d /usr/xpg4/bin; then
		PATH=/usr/xpg4/bin:$PATH
	fi
	df -kP "$1" | tail -1 | awk '{ if ($4 ~ /%/) {print $3} else {print $4} }'
	)
}

MS_dd()
{
    blocks=`expr $3 / 1024`
    bytes=`expr $3 % 1024`
    dd if="$1" ibs=$2 skip=1 obs=1024 conv=sync 2> /dev/null | \
    { test $blocks -gt 0 && dd ibs=1024 obs=1024 count=$blocks ; \
      test $bytes  -gt 0 && dd ibs=1 obs=1024 count=$bytes ; } 2> /dev/null
}

MS_dd_Progress()
{
    if test x"$noprogress" = xy; then
        MS_dd $@
        return $?
    fi
    file="$1"
    offset=$2
    length=$3
    pos=0
    bsize=4194304
    while test $bsize -gt $length; do
        bsize=`expr $bsize / 4`
    done
    blocks=`expr $length / $bsize`
    bytes=`expr $length % $bsize`
    (
        dd ibs=$offset skip=1 2>/dev/null
        pos=`expr $pos \+ $bsize`
        MS_Printf "     0%% " 1>&2
        if test $blocks -gt 0; then
            while test $pos -le $length; do
                dd bs=$bsize count=1 2>/dev/null
                pcent=`expr $length / 100`
                pcent=`expr $pos / $pcent`
                if test $pcent -lt 100; then
                    MS_Printf "\b\b\b\b\b\b\b" 1>&2
                    if test $pcent -lt 10; then
                        MS_Printf "    $pcent%% " 1>&2
                    else
                        MS_Printf "   $pcent%% " 1>&2
                    fi
                fi
                pos=`expr $pos \+ $bsize`
            done
        fi
        if test $bytes -gt 0; then
            dd bs=$bytes count=1 2>/dev/null
        fi
        MS_Printf "\b\b\b\b\b\b\b" 1>&2
        MS_Printf " 100%%  " 1>&2
    ) < "$file"
}

MS_Help()
{
    cat << EOH >&2
${helpheader}Makeself version 2.3.0
 1) Getting help or info about $0 :
  $0 --help   Print this message
  $0 --info   Print embedded info : title, default target directory, embedded script ...
  $0 --lsm    Print embedded lsm entry (or no LSM)
  $0 --list   Print the list of files in the archive
  $0 --check  Checks integrity of the archive

 2) Running $0 :
  $0 [options] [--] [additional arguments to embedded script]
  with following options (in that order)
  --confirm             Ask before running embedded script
  --quiet		Do not print anything except error messages
  --noexec              Do not run embedded script
  --keep                Do not erase target directory after running
			the embedded script
  --noprogress          Do not show the progress during the decompression
  --nox11               Do not spawn an xterm
  --nochown             Do not give the extracted files to the current user
  --nodiskspace         Do not check for available disk space
  --target dir          Extract directly to a target directory
                        directory path can be either absolute or relative
  --tar arg1 [arg2 ...] Access the contents of the archive through the tar command
  --                    Following arguments will be passed to the embedded script
EOH
}

MS_Check()
{
    OLD_PATH="$PATH"
    PATH=${GUESS_MD5_PATH:-"$OLD_PATH:/bin:/usr/bin:/sbin:/usr/local/ssl/bin:/usr/local/bin:/opt/openssl/bin"}
	MD5_ARG=""
    MD5_PATH=`exec <&- 2>&-; which md5sum || command -v md5sum || type md5sum`
    test -x "$MD5_PATH" || MD5_PATH=`exec <&- 2>&-; which md5 || command -v md5 || type md5`
	test -x "$MD5_PATH" || MD5_PATH=`exec <&- 2>&-; which digest || command -v digest || type digest`
    PATH="$OLD_PATH"

    if test x"$quiet" = xn; then
		MS_Printf "Verifying archive integrity..."
    fi
    offset=`head -n 532 "$1" | wc -c | tr -d " "`
    verb=$2
    i=1
    for s in $filesizes
    do
		crc=`echo $CRCsum | cut -d" " -f$i`
		if test -x "$MD5_PATH"; then
			if test x"`basename $MD5_PATH`" = xdigest; then
				MD5_ARG="-a md5"
			fi
			md5=`echo $MD5 | cut -d" " -f$i`
			if test x"$md5" = x00000000000000000000000000000000; then
				test x"$verb" = xy && echo " $1 does not contain an embedded MD5 checksum." >&2
			else
				md5sum=`MS_dd_Progress "$1" $offset $s | eval "$MD5_PATH $MD5_ARG" | cut -b-32`;
				if test x"$md5sum" != x"$md5"; then
					echo "Error in MD5 checksums: $md5sum is different from $md5" >&2
					exit 2
				else
					test x"$verb" = xy && MS_Printf " MD5 checksums are OK." >&2
				fi
				crc="0000000000"; verb=n
			fi
		fi
		if test x"$crc" = x0000000000; then
			test x"$verb" = xy && echo " $1 does not contain a CRC checksum." >&2
		else
			sum1=`MS_dd_Progress "$1" $offset $s | CMD_ENV=xpg4 cksum | awk '{print $1}'`
			if test x"$sum1" = x"$crc"; then
				test x"$verb" = xy && MS_Printf " CRC checksums are OK." >&2
			else
				echo "Error in checksums: $sum1 is different from $crc" >&2
				exit 2;
			fi
		fi
		i=`expr $i + 1`
		offset=`expr $offset + $s`
    done
    if test x"$quiet" = xn; then
		echo " All good."
    fi
}

UnTAR()
{
    if test x"$quiet" = xn; then
		tar $1vf - 2>&1 || { echo Extraction failed. > /dev/tty; kill -15 $$; }
    else

		tar $1f - 2>&1 || { echo Extraction failed. > /dev/tty; kill -15 $$; }
    fi
}

finish=true
xterm_loop=
noprogress=n
nox11=n
copy=none
ownership=y
verbose=n

initargs="$@"

while true
do
    case "$1" in
    -h | --help)
	MS_Help
	exit 0
	;;
    -q | --quiet)
	quiet=y
	noprogress=y
	shift
	;;
    --info)
	echo Identification: "$label"
	echo Target directory: "$targetdir"
	echo Uncompressed size: 36 KB
	echo Compression: gzip
	echo Date of packaging: Wed Apr 11 15:42:10 CDT 2018
	echo Built with Makeself version 2.3.0 on darwin17
	echo Build command was: "./makeself/makeself.sh \\
    \"--notemp\" \\
    \"../../questions/potd3_054_adjMatrix/potd-q54\" \\
    \"../../questions/potd3_054_adjMatrix/clientFilesQuestion/potd-q54.sh\" \\
    \"Extracting potd-q54\" \\
    \"echo\" \\
    \"The initial files can be found in the newly created directory: potd-q54\""
	if test x"$script" != x; then
	    echo Script run after extraction:
	    echo "    " $script $scriptargs
	fi
	if test x"" = xcopy; then
		echo "Archive will copy itself to a temporary location"
	fi
	if test x"n" = xy; then
		echo "Root permissions required for extraction"
	fi
	if test x"y" = xy; then
	    echo "directory $targetdir is permanent"
	else
	    echo "$targetdir will be removed after extraction"
	fi
	exit 0
	;;
    --dumpconf)
	echo LABEL=\"$label\"
	echo SCRIPT=\"$script\"
	echo SCRIPTARGS=\"$scriptargs\"
	echo archdirname=\"potd-q54\"
	echo KEEP=y
	echo NOOVERWRITE=n
	echo COMPRESS=gzip
	echo filesizes=\"$filesizes\"
	echo CRCsum=\"$CRCsum\"
	echo MD5sum=\"$MD5\"
	echo OLDUSIZE=36
	echo OLDSKIP=533
	exit 0
	;;
    --lsm)
cat << EOLSM
No LSM.
EOLSM
	exit 0
	;;
    --list)
	echo Target directory: $targetdir
	offset=`head -n 532 "$0" | wc -c | tr -d " "`
	for s in $filesizes
	do
	    MS_dd "$0" $offset $s | eval "gzip -cd" | UnTAR t
	    offset=`expr $offset + $s`
	done
	exit 0
	;;
	--tar)
	offset=`head -n 532 "$0" | wc -c | tr -d " "`
	arg1="$2"
    if ! shift 2; then MS_Help; exit 1; fi
	for s in $filesizes
	do
	    MS_dd "$0" $offset $s | eval "gzip -cd" | tar "$arg1" - "$@"
	    offset=`expr $offset + $s`
	done
	exit 0
	;;
    --check)
	MS_Check "$0" y
	exit 0
	;;
    --confirm)
	verbose=y
	shift
	;;
	--noexec)
	script=""
	shift
	;;
    --keep)
	keep=y
	shift
	;;
    --target)
	keep=y
	targetdir=${2:-.}
    if ! shift 2; then MS_Help; exit 1; fi
	;;
    --noprogress)
	noprogress=y
	shift
	;;
    --nox11)
	nox11=y
	shift
	;;
    --nochown)
	ownership=n
	shift
	;;
    --nodiskspace)
	nodiskspace=y
	shift
	;;
    --xwin)
	if test "n" = n; then
		finish="echo Press Return to close this window...; read junk"
	fi
	xterm_loop=1
	shift
	;;
    --phase2)
	copy=phase2
	shift
	;;
    --)
	shift
	break ;;
    -*)
	echo Unrecognized flag : "$1" >&2
	MS_Help
	exit 1
	;;
    *)
	break ;;
    esac
done

if test x"$quiet" = xy -a x"$verbose" = xy; then
	echo Cannot be verbose and quiet at the same time. >&2
	exit 1
fi

if test x"n" = xy -a `id -u` -ne 0; then
	echo "Administrative privileges required for this archive (use su or sudo)" >&2
	exit 1	
fi

if test x"$copy" \!= xphase2; then
    MS_PrintLicense
fi

case "$copy" in
copy)
    tmpdir=$TMPROOT/makeself.$RANDOM.`date +"%y%m%d%H%M%S"`.$$
    mkdir "$tmpdir" || {
	echo "Could not create temporary directory $tmpdir" >&2
	exit 1
    }
    SCRIPT_COPY="$tmpdir/makeself"
    echo "Copying to a temporary location..." >&2
    cp "$0" "$SCRIPT_COPY"
    chmod +x "$SCRIPT_COPY"
    cd "$TMPROOT"
    exec "$SCRIPT_COPY" --phase2 -- $initargs
    ;;
phase2)
    finish="$finish ; rm -rf `dirname $0`"
    ;;
esac

if test x"$nox11" = xn; then
    if tty -s; then                 # Do we have a terminal?
	:
    else
        if test x"$DISPLAY" != x -a x"$xterm_loop" = x; then  # No, but do we have X?
            if xset q > /dev/null 2>&1; then # Check for valid DISPLAY variable
                GUESS_XTERMS="xterm gnome-terminal rxvt dtterm eterm Eterm xfce4-terminal lxterminal kvt konsole aterm terminology"
                for a in $GUESS_XTERMS; do
                    if type $a >/dev/null 2>&1; then
                        XTERM=$a
                        break
                    fi
                done
                chmod a+x $0 || echo Please add execution rights on $0
                if test `echo "$0" | cut -c1` = "/"; then # Spawn a terminal!
                    exec $XTERM -title "$label" -e "$0" --xwin "$initargs"
                else
                    exec $XTERM -title "$label" -e "./$0" --xwin "$initargs"
                fi
            fi
        fi
    fi
fi

if test x"$targetdir" = x.; then
    tmpdir="."
else
    if test x"$keep" = xy; then
	if test x"$nooverwrite" = xy && test -d "$targetdir"; then
            echo "Target directory $targetdir already exists, aborting." >&2
            exit 1
	fi
	if test x"$quiet" = xn; then
	    echo "Creating directory $targetdir" >&2
	fi
	tmpdir="$targetdir"
	dashp="-p"
    else
	tmpdir="$TMPROOT/selfgz$$$RANDOM"
	dashp=""
    fi
    mkdir $dashp $tmpdir || {
	echo 'Cannot create target directory' $tmpdir >&2
	echo 'You should try option --target dir' >&2
	eval $finish
	exit 1
    }
fi

location="`pwd`"
if test x"$SETUP_NOCHECK" != x1; then
    MS_Check "$0"
fi
offset=`head -n 532 "$0" | wc -c | tr -d " "`

if test x"$verbose" = xy; then
	MS_Printf "About to extract 36 KB in $tmpdir ... Proceed ? [Y/n] "
	read yn
	if test x"$yn" = xn; then
		eval $finish; exit 1
	fi
fi

if test x"$quiet" = xn; then
	MS_Printf "Uncompressing $label"
fi
res=3
if test x"$keep" = xn; then
    trap 'echo Signal caught, cleaning up >&2; cd $TMPROOT; /bin/rm -rf $tmpdir; eval $finish; exit 15' 1 2 3 15
fi

if test x"$nodiskspace" = xn; then
    leftspace=`MS_diskspace $tmpdir`
    if test -n "$leftspace"; then
        if test "$leftspace" -lt 36; then
            echo
            echo "Not enough space left in "`dirname $tmpdir`" ($leftspace KB) to decompress $0 (36 KB)" >&2
            echo "Use --nodiskspace option to skip this check and proceed anyway" >&2
            if test x"$keep" = xn; then
                echo "Consider setting TMPDIR to a directory with more free space."
            fi
            eval $finish; exit 1
        fi
    fi
fi

for s in $filesizes
do
    if MS_dd_Progress "$0" $offset $s | eval "gzip -cd" | ( cd "$tmpdir"; umask $ORIG_UMASK ; UnTAR xp ) 1>/dev/null; then
		if test x"$ownership" = xy; then
			(PATH=/usr/xpg4/bin:$PATH; cd "$tmpdir"; chown -R `id -u` .;  chgrp -R `id -g` .)
		fi
    else
		echo >&2
		echo "Unable to decompress $0" >&2
		eval $finish; exit 1
    fi
    offset=`expr $offset + $s`
done
if test x"$quiet" = xn; then
	echo
fi

cd "$tmpdir"
res=0
if test x"$script" != x; then
    if test x"$verbose" = x"y"; then
		MS_Printf "OK to execute: $script $scriptargs $* ? [Y/n] "
		read yn
		if test x"$yn" = x -o x"$yn" = xy -o x"$yn" = xY; then
			eval "\"$script\" $scriptargs \"\$@\""; res=$?;
		fi
    else
		eval "\"$script\" $scriptargs \"\$@\""; res=$?
    fi
    if test "$res" -ne 0; then
		test x"$verbose" = xy && echo "The program '$script' returned an error code ($res)" >&2
    fi
fi
if test x"$keep" = xn; then
    cd $TMPROOT
    /bin/rm -rf $tmpdir
fi
eval $finish; exit $res
‹ "sÎZí<pÅ•³ZÉ’×Ÿ•Iîğ1@¶±vWükâU,Áb[90ãÕîHZgìÎÚr‚%òR/İåœ@ğQ¢êê0©ËîçT	rdp!]ê+á@IùR#ÊIT&q¬{¯»gwvv%ÛÂÇÕviºûu¿÷úõë×¯{¶5]nã®y°ÛíµÕÕ<Ikhj¯¨¢)¼£²¢¦²ªÖQe¯æíGUe5ÇWssbQÙQ$¯×'Í€hmm3wû‘J?#¡ÜÖìşªÔæóK×tükªª¦ÿê
‡6şöªÊZÿJ{m-ÇÛóãÍƒ°[¸{K³Àoæn_ĞrÏ]_ÚÉòå!ŞíİçöHAÏA1à–#¾ÎòÅ²u÷n@ğøİÁöuëúbÓ–mH³>*{7{Ö­³wòë=üúv~ı=v~ıŸ»ı~KSƒ¤©!›Âb´üª2&Î‹%Çr”keşª²¦†5üªyÌ6køõ!~U½ÅBEŞ@E÷„ÃˆÒö&ä(:´”Õ©Ù‰Ç/¹ƒ,ó×Güú6~-(J'8÷Ù
å¶¬¾wÌõü¯©­1Ìÿjğ ùù?a¥/èñÇ¼¿ÉŠÊÉ¸Ã’.ƒ_°ıµ½R/nÛ±Åå-+ğ¥‹t¤h¬‰‡™½ÑÔ1Ìo‹¸Ãü×-<­¡íZœsÍÄà6¦j|A™n´tm´X‰÷„‚2Lç¨àm—Ê((ŠÊüZ–¶ßFh¢Íx¥¨¼¨÷‡|^hÀ›&\;-.i4¸'&·‡ H©±ı:¢Æ '¸l""U4*ÌÜøˆÈ$öÀpÈÒ—¥ˆìó {Öæ—c¥ôúÚ,×dşƒï›Óù_á¨¨0Îÿªªüúÿkş³¢ÙËÅ
Kîéo±­µğ°_
H`ºr‡Ä·Å‚Ù†Î·JşĞ¿Öö‰¦:ó*IE‚|›Û•À}\Áüˆå> ³e;iu^!›Ës:‡ê	Åd~Ó&~ÅmlxêY7ğ+°|…?ídÛBølÁì!ÙÄ·¯¿#ˆ¹u›y‡sÍ>J³/E³oZ½»çƒ’¯½£¸lFº”ë¿ß·çş}{6NK«õ,E½äWdtY¦#]ñ@ğ ¢U|	Ëzc³ñw‡ mÉËË!>{ÌØMÙ1AÓìú”P°lm•„åÀ5Uê=# ëáıˆÂ4‘k„‡#dĞ¦¥ãtcH¹æÃìq"t–L•wéí·í—Ë‡ÏÔş_˜>Íßªj`á7¬ÿ•ùıÿÜ„Ó¿ÿÍÅb3™9.^ p)<cE—ˆÔÿ<NxDÑµe›pŸ°ãl¦Ëhùt Ÿ]Âî]9èc0ÈÆ±”#rÈR§œF3òÛ»ˆòûÃâ4\¨ç[ÈÅõ (FåXktZ~,£ü¾e€µPÂøÍËà'vHş°ÉÁïFJÌ(_»Ç#J),‹²»ÕÈoğ&Êo\Ì0¢è¡›ºiúûÔrÊïk:X
²øÅ‚|A¯è¶…rğûüÍ”_Y>©Cl‹ÀN2·|ıŒßnl¾„İÕ™OÃ–][2í‹Œci¦½i0Ú›™KóÅ _Œ´†übXLË¯TëåšÇdÑkHÎ€3ù•2~Ku°‘ßô~÷Lò•1~£:XÏû»Ü 7:ï›ïŞ.44jc`7ÌWæ¤Ş”æ£Ÿ+À¯Ø^	ÎHû!=énD¬€*Ä^Íqª‰ÚöÙ^Kmğïïä¸I˜Ã ïb<[ØÅ\«NG¹ÂØW›£|1ëÂBxl±hÄæ÷µÚ¼ı^V#ã{ú÷üè'5g¿·ãæë—X°JãEò¬Õ9êú‹tÃK2Û[ÈTµT›CóûL¨©TÛğxÖ­+w”ƒ¾V‚_§ÇçÎñwŒÊR ü.FBÂ-L¥µ”vƒíµù½N>äC>äC>äÃ\…§òßÎîsõÎdÑs×Á–ª·zìœJ—êTFÔz+)ºŞ„EÂy§›t&»ÎOµLg‘JÇ’ÂYg²Yİ¸¥ÄÚû”d³ŸÆSÅ;5¹ùNl!)¨Î$çô4Uá(¤ã‡?_ÿ•©§åİÃ-ïBÆúÃ÷Õ“†_(Å&FÔÁÅ¤©QàÙ=lR„ñ‰g§¦¦€ão–¼nÀS¿I‘†A‚tv¢‘ºOÕïy¹Ízó!}yèaª;Î$¦Â$ë5«¡¨ 'cãNå%u92ï>Å+åü=×GtÑ5$ğ;“õÀéc§é¤SiEx×Bøô
c€4@Ú©=¤á *m«z5ôGİ·ˆt¤8,¼jì>ç{„œÉ·Â™Ü_g=Ñõ‘ÒuÔ©¼Ù ‰õ5(¯@}ßÔk=Â“¸Atöv)?CÅ)Í§T7íÒ¨Z¸ˆV©÷Ğ’ÃÂ“Ö“ê2
A\ı“èD5*çÂn”ôY?ˆ›8å¤ÒuÁ©œVo@â‘“@7õ#ÜU+o:>üb4*Ã ñ&¢3Ğ©1•£Z @˜LÆ.@g'•‡Jœ³ãÎ~¹ÅÚw$sZëj¤¡§ù´úı…HÔCfÎQs;·Q¸`ía¿G†±i°­Îî9§µi¥ú(9ßéé:çT
?TJ¬âÕÄNÁˆcª¾RX‚(ÄªÔ‰ÓÀz‚5IŠÎM<B¬è¨±õ¶¦Ä>PZ˜µİIÜ#| mŒ€4“‡ÖÅ	¶r–ô¯Ö™ö¼Ü§ÃŸ¦Ïˆğ6GfêÚ3 –ód¦–SŒC·Ï|XXb²šO0Ş&=}›õûÎL¼ı†:Ãür3	4yºÏ¹~‚/sê«œšK0QÏ­vQo 'ğ‚cQ1Q'dü™åÅÔôÃø©ßƒ÷  R¿}qjjèõ3¦,ˆ½¹õZ-Ì,°7ìÿT$›_$¶_ıØBêPÔù2s¼ÆØ«N»au9e÷9ÊœF—æD•1uQACÊÄ‰Çr’™s¼$GêH·“ÂŞ¡2HÔã ]¢Sgo¢Œ`%\‹$‡KzzQ‡3YøSÓÄ·Ì¿„ öÒ6{IÒ»›Rï¥u»i]˜ÖuĞº0­£Ğgr‡ê4“èí¤åt¶7ÇÁ$ài,s*% t=¥%%¡ ˆFÑ‹¡Kv–@/ÏƒÚ”¡ N&0^= ÂvŸç­‰È`%s¢
¥ÊÄb]-ˆ&½½—jím=´ü¾¦ÆäÎä#Ppü8Şº«ñ‹¤·gHrdŒ$ÇG!QW–`k½˜Oşú2XZæCYJ³ÿEÅ%4:n:n0«·ŠSC…n<å²#ğê¯şK1wbø"sÄk÷>9÷Ğ‹¤|w‚æ@B<øÌOü#r[‡bj>œòt1™¢£I€•D?j½÷(¢–š˜_? ‡	¹õÄ¨39„ıQìKw.Ùƒ@ÏO{<Ct=TB¬BBD§CºlJ&P—êabÚ‰>­«`‰¥Ôrú¨â	¤ ã€³»³&ÆiËYÆÓ7"<^DN_²åq\ó”–£ õ']|2vtşûàï•7ÔïÏcª§Ñ_2•$ø)ìû2ôÇ´2>ˆ¯J¥]Àr6ë@FB™@Jµn
0„%jÍSA­TĞÓÀ´;~«&…ºm^º?Â¹¾çe¦:­ÏB¯&ŞÂ62ÅTeôDk6ZD…Qz‘– ¶Ş×­¯¥d¿nG}³ˆlDJ÷¼¬÷7c CCètˆÚîqjzGé E5)‰cô$Ç²n‚có9½É“8THŠ¨â§5À:ñÇ(ê
Ñ¤~)3	(!ÉF‚m}â&Ê’RY›"’›X@ÖjB@Õ‘4¶ÙC…í!Â>ñu°+y¤‡ÊÃÓ‘]J}-5IZHLn/õRêJ¨Lı¤Dbë¡åTÑ´ z}Èfm¼º‡ú¨ à*4ˆ¿`)/9{{ÃtamÃaªa…z.eÈICOÂF'	ug´ğ0-Tz]ÓI1·QL—Ş£QW>D;M”!êÃIÒ¨|Ü¤ô2O¿­èy\&4?9´›xƒ#È[½•, Lèã´«9İ¿ÒÜF×Æ×i¤OAú”õ‡Åİã¦îóà»(µ“èÑÆ&:õŸàGšGq)xÌfÜeÀÓÆÜ[‘MÚ4z?ÛçÂ’Òî“&Èƒo‡Gi†t˜‘ãºÁ5Ù±0óJ¡"¿UpOßN–mÍp«ß¢ÒáWušŸ`ó‘€Yô‘î8g2‹‘Ä^ˆqë0±–N}Ô'™ğÄyq^„‰­Qt.©>oB÷„¥‰a%QEZ¼Y%I
ºU“’°“šßaÍ<SZR6=/kªfL•j%*ÔÕ³í„6U8:UHİL•F*™Ÿ˜Äd„,ê¬‚”$à2zÙúBV|J2¸9éŠ.vŒ¸X´tT¸=NÛoºÔÍdá*õiÈı&¼„(±óğ.±1vŞú×Ã¸•ÖïÏñå ö¦S«_‰0µz”Ä?%ñ¿‘øU‘ø5ÿœÄoø‰ß"ñÛ$~‡Äã$ş‰Ï’ø×$VIü‰Ï‘ø·$$ñû$ş€Ä’#¦Ş›AfËÎ©Õ?Ó×ñ§¨Õ_ƒĞ[1÷6æğ€lâ5Ìbnsf4˜«ÇÜ0÷æ¾‹¹S˜ëÇ\qî"	?Ì=¹½˜ûæú1‡n ãeÌ1÷Ìy1÷?˜»sdÚ “ê@i'¾ˆ¹e˜Û„ú}fùVÒí—Ì yüGğèF«™zæÑGãqÎ»•Dt —¢ÿ qñİãa~Àı¸BëæCô¬Vósú0{İLXñHõ] îÔj–ÒsÚÛ1]YH"´óİ.“ƒÉ\«<a©,ØÑ‹ÚÙQö¿2q<g±pø?U|[$àíøO9^êôEå(_&u†%,yùPLÇdŞåk€*Mà@ûLv$ çWËØ¹Ë*—¥Ï>¼—çéÙ2CÖ³saKö/¦ç°×qÜYHùë9Î¿œ·-äJ¸ÛØ	‘©˜ºŠ8S?Åğ|Î4¼˜+[LÎ~È±õç5~mgê,5•.,.Á“£e©rW©)~ÄJõõ›°S$XµÁØ²¸t6¿c}‡3²t1Kíì<Û¥‹_ÏÎ-Yı
–ŞÆÒ–Ö³ôK,İÅRÔ¦2¸ƒ¥–ºÓôá›rÿßÁc¬¼¥ËÒ¸éòşO!ò!ò!ò!ò!ò!ò!òáÿcX²¢lïçîÚÅ-©ÅûîŞ)WŠ¢£
¿¸{#¥}ºÒj|0,5z„
ŸWàHMĞàâı¦şóS,˜hñ g(Û;;Å°‰†‚n¿O>(î·sqüñ‘¡\Wÿ{Kÿw¾ÕôJ.~&Íãt‚8wD”#nŸyé!Ñ”E”Oğù@ğú¾ğ+¥P›°Ÿ›	¥Æ/ÛåÁµİ“WQ%Šá˜,"¶Û#K1*=“‚ĞËİ;E#;°¸²ÕõyDöeã.q—]„µ¢kûÎ*tX§oÄm¼|^]È©FDIãKÓ£Rr©Aaã)»j[èÿğï¢±€Äõq;µ‡íT¢ÈãöK·Ç¢’ØæöH2Šb©!p‘ºä:_(*‚RM»$!ªš‹ìÖ˜£‚v‚~‰[uPv»ÛË¡é£ì“…€îıäT¶VFç‹J°È¾râê6,ÚAÛî¼ròš¨”#·:„;+Ed7Íš‹¦ƒY°ğG…V¤}*ƒ–‹Æ¸?£2e• E%òŞVŸ,ºÁúğËVÌ|1"É‘ĞJ,7}5ŞÂö!Ê EÙ)ß32àºe‰‘rÑÓé[¥v_P„¡÷t`Í™t„-kåã&˜) ’!¨"* oÀiÆ$rE\È-!ßÍ5sQê”<1Y;$7º®5÷qëÉÅfçb’sÏ™jÙÉ®jò(xªÿ,à
íéO}]ÛiwÔT•ùmª;]P¡ÿFZC&?/¨	ÆÜ;fnühsÖşã§×È¿=g6Rrg6-ãÌø™şDñWñ	vª“Ü_™3>ªNW|ÛlæV õÀJäQáÀjP‚Şƒ¿fÎZ"Ş4³Uàm#^TuÕiëâø“&W¼ ¿ Ş7•NšÆJø¸¥”ç~YÈq^İú¹«(Şp›ÙÄí(ûæ}ÚaA¹)ı]ÚP¿JÿàC:úß`;ƒ˜ÔÁw›2aä¯ı.îGùk°Fï2Ğ»ôıÿyF¯Á¿™—)ÏÎâÜòi?®küõ0áo¥ùÅŒ?ƒ_Ÿ†oàÇø—Òü{ŒŸkôõ†şÖúË-¡ùy%”^ƒ5z»Şn ·3ü½İ@7ĞÇô£0Ğ—èKôÃÿ+Œ~Ø@){ÑÎYD­}—äW¡Á~
údç>ëæ3}2øŞùÓŒ¯A>Ş Ïè_dü4ØnÉä÷´.Z	ğ›¦±ß"ƒıäaç[ÂB&Ï²+/íœl£·è/¥í|­‡Ñ»ôôüàßR˜şNôÇ>Ëàg°yšŸWwP6õËMñ”?¸e…µñø3yşàÅºöN1Xkïkíá2iÕùÃ¬Íçkşb;ƒµù²ÁÚüS¬Íçg	Oéû_¬éï=köŠÿã‡°6b°æO@àÒ”®GML^ÖŞ«ä«¾ô'®ïP8¥Ñ¢‚Lx¥àïà÷ŒôæL¸Õ ?k€_7Àgğ|]a&¼İ ·àNÇSğ#ø1şÓø»x¸0SŸ£xÌ ãg®Éøàœƒ?'ÆÏ÷ñß:ğ?+|Ìv¾
Ïƒ¨wxğûe¼úÿqâ!|7‚¿ÇïtñßŞ`ÿà9H›ª÷C€gÿµËs°ËuäÚ/bùtÛC¬ËÚMbañªšô®J«f7ÕÙ{Ù+xA¿ä»ùå½™_ó÷ò«ğN.ŠÙ¯"yu0¾u_æûöÌoÚWñûª¼]Ïò½zÖoÔŸğ]ú“¼DÏêí9ëG·é~oËı~=ëë™Ş¨³^¥ÙoJìG¤oÔY¯Ò¹ğÓŞ¬µWê¬Ÿül-Q °	x‹­!ä‰áÅaQÛÖP Ìäßéñáœ¶µ4¶lµí‹}¡H…Í*ø¤&>hû×{¢Õ6˜ıQrÓ˜-’½•¢½ºVL]gDÊÖ?T]kã´ëB?…¶é¥Óúôr[Löù£T¸-á°ßº%\w{B^©ÜƒtA™ˆÙ í—ü!Ğ·mW(äSóixR›;æ—Ë;=²VCî`ºÙ<ëÖÙö;lÌV¯ËKÙ4Ş<6;<]ÃŸƒVD‘Î×Üëğ4×/^b=Ï¹ø^bÁq¹gÒ5^Nç`HAV\€.s1£ÏnÛÑ	klØí%™şëétJÅÒÛ™6[cm¢Ö»”fk¨fwÖáAÊ™P<İc¦}íJ)anÚÂ»öÁrŠœƒ†p#Á,‹¬Õx7¬Ô~¿˜Zå¯êö}ÛÖ­º+«üREVÉíY%ÇUZ8>™6Rg©ˆ9ôş¿«ç¯>\âş_‡½¦"ûşÿüısÒwız¢²×Òßş»_·™åí¿xÓ(šV™î6Õ­xw-¹¼7ÆÛS×±V²;Tµ{|Ûoãí·ñìRTd—VP\­mî°É%Än~H±î’İv5uoì•~iB®ÚÍ¸ÆX8óŞ\úË”i }¿d—õéKÎ*2ÈßïšùùùùùùùùğÙ
ÿ˜fNl x  