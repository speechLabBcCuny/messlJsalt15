for ((i=1; i<=20; ++i));
do 
	matlab -r "create_messl_soft_masks($i,20)" & 
done
wait;
echo "done"
